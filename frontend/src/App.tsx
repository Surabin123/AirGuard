import React, { useEffect, useState, useMemo } from 'react';
import { FixedSizeList as List } from 'react-window';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, LineChart, Line
} from 'recharts';
import create from 'zustand';

// Cesium and Resium imports
import { Viewer, Entity, PointGraphics, PolylineGraphics, EllipseGraphics } from 'resium';
import { Cartesian3, Color } from 'cesium';
import "cesium/Build/Cesium/Widgets/widgets.css";


const BACKEND_PORT = '8001';
const API_BASE = `http://127.0.0.1:${BACKEND_PORT}`;
const WS_BASE = `ws://127.0.0.1:${BACKEND_PORT}`;

// --- Types ---
interface TrailPosition {
  lat: number;
  lng: number;
}

interface Flight {
  id: string; // ICAO24
  callsign: string;
  squawk?: string;
  emergencyCode?: string;
  rapidDescent?: boolean;
  routeDeviationNm?: number | null;
  altitude: number; // ft
  speed: number; // knots
  heading: number; // degrees
  trustScore?: number; // 0-100 when analysis is available
  signalStrength?: number; // dBm, only when supplied by receiver
  status: 'normal' | 'suspicious' | 'critical' | 'unknown';
  lat: number;
  lng: number;
  history?: TrailPosition[];
  trilateration?: string;
  ruleFlags?: {
    positionJump: boolean;
    duplicateIcao: boolean;
    climbRate: boolean;
    altVelMismatch: boolean;
  };
  shapValues?: { name: string; value: number }[];
}

interface AlertLog {
  id: string;
  timestamp: string;
  callsign: string;
  icao24: string;
  type: string;
  severity: 'low' | 'medium' | 'high';
  scoreImpact: number;
  acknowledged: boolean;
  riskScore?: number;
  ruleFlags?: string[];
  modelScores?: { ensemble_score?: number; autoencoder_score?: number };
  evidence?: Record<string, unknown>;
}

interface GeofenceRecord {
  id: number;
  name: string;
  latitude: number;
  longitude: number;
  radius_km: number;
  enabled: boolean;
}

interface HealthStats {
  poll_latency_ms: number;
  queue_depth: number;
  circuit_breaker_state: string;
  last_successful_poll: string | null;
}

interface ModelRunStats {
  id: number;
  run_at: string;
  model_version: string;
  true_positives: number;
  false_positives: number | null;
  true_negatives: number | null;
  false_negatives: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  notes: string;
}

interface RecordedState {
  id: number;
  icao24: string;
  callsign?: string | null;
  latitude: number;
  longitude: number;
  altitude_m: number;
  velocity_ms: number;
  heading_deg: number;
  vertical_rate_ms: number;
  on_ground: boolean;
  squawk?: string | null;
  received_at: string;
  source: string;
}

interface IncidentRecord {
  id: string;
  icao24: string;
  status: string;
  created_at: string;
  reason?: string;
  rule_flags?: string[];
  comments?: Array<{ at: string; text: string }>;
  timeline?: Array<{ at: string; type: string; signals: string[]; squawk?: string | null; risk_score?: number }>;
}

// --- Zustand State Management ---
interface IngestionPayload {
  icao24: string;
  latitude: number;
  longitude: number;
  altitude_m: number;
  velocity_ms: number;
  heading_deg: number;
  callsign?: string;
  trust_score?: number;
  risk_score?: number;
  status?: 'live' | 'stale' | 'lost';
  rule_flags?: string[];
  received_at?: string;
  source?: string;
  squawk?: string | null;
  emergency_code?: string | null;
  rapid_descent?: boolean;
  route_deviation_nm?: number | null;
  events?: Array<{ at: string; type: string; signals: string[]; squawk?: string | null }>;
}

interface AirGuardState {
  flights: Flight[];
  selectedFlightId: string | null;
  alerts: AlertLog[];
  backendHealth: 'online' | 'offline' | 'checking';
  websocketStatus: 'connecting' | 'connected' | 'disconnected' | 'reconnecting';
  activeFilter: 'all' | 'suspicious' | 'critical';
  setFlights: (flights: Flight[]) => void;
  updateFlightStatus: (icao24: string, score: number) => void;
  updateOrAddFlight: (payload: IngestionPayload) => void;
  setSelectedFlightId: (id: string | null) => void;
  setBackendHealth: (status: 'online' | 'offline' | 'checking') => void;
  setWebsocketStatus: (status: 'connecting' | 'connected' | 'disconnected' | 'reconnecting') => void;
  setActiveFilter: (filter: 'all' | 'suspicious' | 'critical') => void;
  addAlert: (alert: AlertLog) => void;
  acknowledgeAlert: (id: string) => void;
}

const useStore = create<AirGuardState>((set) => ({
  flights: [],
  selectedFlightId: null,
  alerts: [],
  backendHealth: 'checking',
  websocketStatus: 'connecting',
  activeFilter: 'all',
  setFlights: (flights) => set({ flights }),
  updateFlightStatus: (icao24, score) => set((state) => {
    const updated = state.flights.map(f => {
      if (f.callsign.toLowerCase() === icao24.toLowerCase()) {
        const scorePercentage = Math.round(score * 100);
        const status: Flight['status'] = score >= 0.8 ? 'critical' : score >= 0.4 ? 'suspicious' : 'normal';
        return { ...f, trustScore: 100 - scorePercentage, status };
      }
      return f;
    });
    return { flights: updated };
  }),
  updateOrAddFlight: (payload: IngestionPayload) => set((state) => {
    const existingIndex = state.flights.findIndex(f => f.id === payload.icao24);
    const lat = payload.latitude;
    const lng = payload.longitude;
    const altitude = Math.round(payload.altitude_m * 3.28084);
    const speed = Math.round(payload.velocity_ms * 1.94384);
    const heading = Math.round(payload.heading_deg);
    
    if (existingIndex > -1) {
      const updated = [...state.flights];
      const existing = updated[existingIndex];
      const prevHistory = existing.history || [];
      const newHistory = [{ lat: existing.lat, lng: existing.lng }, ...prevHistory].slice(0, 5);
      
      updated[existingIndex] = {
        ...existing,
        lat,
        lng,
        altitude,
        speed,
        heading,
        trustScore: payload.trust_score ?? existing.trustScore,
        squawk: payload.squawk ?? existing.squawk,
        status: payload.status === 'lost' || Boolean(payload.emergency_code) || payload.rapid_descent ? 'critical' :
          payload.status === 'stale' || (payload.route_deviation_nm ?? 0) > 10 ? 'suspicious' :
          (payload.risk_score ?? 0) >= 0.8 ? 'critical' : (payload.risk_score ?? 0) >= 0.4 ? 'suspicious' : 'normal',
        ruleFlags: {
          positionJump: payload.rule_flags?.includes('rule_position_jump') ?? false,
          duplicateIcao: payload.rule_flags?.includes('rule_duplicate_icao') ?? false,
          climbRate: payload.rule_flags?.includes('rule_climb_rate') ?? false,
          altVelMismatch: payload.rule_flags?.includes('rule_alt_vel_mismatch') ?? false
        },
        history: newHistory
      };
      return { flights: updated };
    } else {
      const newFlight: Flight = {
        id: payload.icao24,
        callsign: payload.callsign?.trim() || payload.icao24.toUpperCase(),
        squawk: payload.squawk ?? undefined,
        altitude,
        speed,
        heading,
        trustScore: payload.trust_score,
        status: payload.status === 'lost' || Boolean(payload.emergency_code) || payload.rapid_descent ? 'critical' :
          payload.status === 'stale' || (payload.route_deviation_nm ?? 0) > 10 ? 'suspicious' : 'normal',
        lat,
        lng,
        history: [],
        trilateration: undefined,
        ruleFlags: {
          positionJump: payload.rule_flags?.includes('rule_position_jump') ?? false,
          duplicateIcao: payload.rule_flags?.includes('rule_duplicate_icao') ?? false,
          climbRate: payload.rule_flags?.includes('rule_climb_rate') ?? false,
          altVelMismatch: payload.rule_flags?.includes('rule_alt_vel_mismatch') ?? false
        },
        shapValues: []
      };
      return { flights: [newFlight, ...state.flights] };
    }
  }),
  setSelectedFlightId: (id) => set({ selectedFlightId: id }),
  setBackendHealth: (status) => set({ backendHealth: status }),
  setWebsocketStatus: (status) => set({ websocketStatus: status }),
  setActiveFilter: (filter) => set({ activeFilter: filter }),
  addAlert: (alert) => set((state) => ({ alerts: [alert, ...state.alerts] })),
  acknowledgeAlert: (id) => set((state) => ({
    alerts: state.alerts.map(a => a.id === id ? { ...a, acknowledged: true } : a)
  }))
}));

// --- Cesium Error Boundary ---
class CesiumErrorBoundary extends React.Component<{ children?: React.ReactNode }, { hasError: boolean; error: Error | null }> {
  public state = {
    hasError: false,
    error: null as Error | null
  };

  public static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("Cesium render crash:", error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="w-full h-full bg-[#030712] border border-cyan-950/60 rounded p-6 flex flex-col items-center justify-center text-center font-mono">
          <span className="text-rose-400 text-sm font-bold mb-2">▲ 3D RENDER ENGINE FAILURE</span>
          <p className="text-[10px] text-slate-400 max-w-md leading-relaxed mb-4">
            WebGL context initialization failed or Cesium assets are unreachable. The ground station is operating in fallback list-only mode.
          </p>
          <div className="text-[9px] text-slate-600 bg-black/40 border border-cyan-950/30 p-3 rounded text-left w-full max-w-sm overflow-x-auto">
            {this.state.error?.toString()}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// --- App Component ---
export default function App() {
  const { 
    flights, selectedFlightId, alerts, websocketStatus, activeFilter,
    setSelectedFlightId, setBackendHealth, setWebsocketStatus, setActiveFilter, addAlert,
    updateFlightStatus, updateOrAddFlight, acknowledgeAlert
  } = useStore();

  const [activeView, setActiveView] = useState<'about' | 'dashboard'>('about');
  const [dashboardTab, setDashboardTab] = useState<'radar' | 'alerts' | 'playback' | 'analytics'>('radar');
  const [simulatedTime, setSimulatedTime] = useState<string>('');
  
  // RuleConfig Slider Config State
  const [config, setConfig] = useState({
    max_implied_speed_kmh: 1200.0,
    duplicate_icao_dist_km: 50.0,
    max_vertical_rate_ms: 50.0,
    max_ground_altitude_m: 100.0,
    max_ground_speed_ms: 77.0,
    min_flight_speed_ms: 20.0
  });

  // Replay results
  const [replayResult, setReplayResult] = useState<ModelRunStats | null>(null);
  const [isReplaying, setIsReplaying] = useState<boolean>(false);

  // Historical Playback States
  const [playbackIndex, setPlaybackIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<1 | 5 | 20>(1);
  const [replayRecords, setReplayRecords] = useState<RecordedState[]>([]);
  const [replayMessage, setReplayMessage] = useState('Choose an aircraft with stored observations to load its real history.');
  const [incidentRecords, setIncidentRecords] = useState<IncidentRecord[]>([]);
  const [geofences, setGeofences] = useState<GeofenceRecord[]>([]);
  const [evaluationRuns, setEvaluationRuns] = useState<ModelRunStats[]>([]);
  const [selectedEvidence, setSelectedEvidence] = useState<AlertLog | null>(null);
  const [geofenceForm, setGeofenceForm] = useState({ name: '', latitude: '', longitude: '', radius_km: '5' });
  const [geofenceMessage, setGeofenceMessage] = useState('');

  // Alerts sorting/pagination local state
  const [sortField, setSortField] = useState<'timestamp' | 'callsign' | 'scoreImpact'>('timestamp');
  const [sortAsc, setSortAsc] = useState<boolean>(false);
  const [alertPage, setAlertPage] = useState<number>(0);
  const alertsPerPage = 10;

  const [healthData, setHealthData] = useState<HealthStats>({
    poll_latency_ms: 0,
    queue_depth: 0,
    circuit_breaker_state: 'UNKNOWN',
    last_successful_poll: null
  });

  // Fetch initial thresholds config
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/config`);
        if (res.ok) {
          const data = await res.json();
          setConfig(data);
        }
      } catch (err) {
        console.error("Config fetch failed:", err);
      }
    };
    fetchConfig();
  }, [dashboardTab]);

  // WebSocket Connection Handler
  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimeout: number | null = null;
    let reconnectDelay = 1000;

    const connect = () => {
      setWebsocketStatus('connecting');
      socket = new WebSocket(`${WS_BASE}/api/v1/stream`);

      socket.onopen = () => {
        setWebsocketStatus('connected');
        setBackendHealth('online');
        reconnectDelay = 1000;
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.event === 'ALERT_TRIGGERED') {
            addAlert({
              id: String(data.alert_id ?? Date.now()),
              timestamp: data.detected_at ?? new Date().toISOString(),
              callsign: data.icao24.toUpperCase(),
              icao24: data.icao24,
              type: data.reason_text,
              severity: data.combined_risk_score >= 0.8 ? 'high' : 'medium',
              scoreImpact: -Math.round(data.combined_risk_score * 100),
              acknowledged: false,
              riskScore: data.combined_risk_score,
              ruleFlags: data.rule_flags,
              modelScores: data.model_scores,
              evidence: data.evidence
            });
            updateFlightStatus(data.icao24, data.combined_risk_score);
          } else if (data.event === 'AIRCRAFT_UPDATE') {
            updateOrAddFlight(data.payload);
          } else if (data.event === 'AIRCRAFT_STATUS_CHANGED') {
            updateOrAddFlight(data.payload);
          }
        } catch (err) {
          console.error("Failed to parse websocket message:", err);
        }
      };

      socket.onclose = () => {
        setWebsocketStatus('reconnecting');
        reconnectTimeout = window.setTimeout(() => {
          reconnectDelay = Math.min(reconnectDelay * 2, 30000);
          connect();
        }, reconnectDelay);
      };

      socket.onerror = () => {
        socket?.close();
      };
    };

    connect();

    return () => {
      if (socket) socket.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, [addAlert, updateFlightStatus, updateOrAddFlight, setBackendHealth, setWebsocketStatus]);

  // Check system stats
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/system-health`);
        if (res.ok) {
          const data = await res.json();
          setHealthData({
            poll_latency_ms: data.poll_latency_ms,
            queue_depth: data.queue_depth,
            circuit_breaker_state: data.circuit_breaker_state,
            last_successful_poll: data.last_successful_poll
          });
        }
        const [aircraftResponse, incidentResponse, alertResponse, geofenceResponse, evaluationResponse] = await Promise.all([
          fetch(`${API_BASE}/api/v1/live-aircraft`),
          fetch(`${API_BASE}/api/v1/incidents`),
          fetch(`${API_BASE}/api/v1/alerts?limit=100`),
          fetch(`${API_BASE}/api/v1/geofences`),
          fetch(`${API_BASE}/api/v1/model-runs?limit=20`)
        ]);
        if (aircraftResponse.ok) {
          const currentAircraft = await aircraftResponse.json() as IngestionPayload[];
          currentAircraft.forEach(updateOrAddFlight);
        }
        if (incidentResponse.ok) setIncidentRecords(await incidentResponse.json() as IncidentRecord[]);
        if (alertResponse.ok) {
          const currentAlerts = await alertResponse.json() as Array<{
            id: number; detected_at: string; icao24: string; reason_text: string;
            combined_risk_score: number; acknowledged: boolean; rule_flags: string[];
            shap_explanation: { shap?: Record<string, number>; evidence?: Record<string, unknown> };
            ensemble_score: number; autoencoder_score: number;
          }>;
          useStore.setState({ alerts: currentAlerts.map((alert) => ({
            id: String(alert.id),
            timestamp: alert.detected_at,
            callsign: alert.icao24.toUpperCase(),
            icao24: alert.icao24,
            type: alert.reason_text,
            severity: alert.combined_risk_score >= 0.8 ? 'high' : 'medium',
            scoreImpact: -Math.round(alert.combined_risk_score * 100),
            acknowledged: alert.acknowledged,
            riskScore: alert.combined_risk_score,
            ruleFlags: alert.rule_flags,
            modelScores: { ensemble_score: alert.ensemble_score, autoencoder_score: alert.autoencoder_score },
            evidence: alert.shap_explanation?.evidence ?? alert.shap_explanation
          })) });
        }
        if (geofenceResponse.ok) setGeofences(await geofenceResponse.json() as GeofenceRecord[]);
        if (evaluationResponse.ok) setEvaluationRuns(await evaluationResponse.json() as ModelRunStats[]);
      } catch (err) {
        // Degrade gracefully
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 5000);
    return () => clearInterval(interval);
  }, [updateOrAddFlight]);

  // Playback timer loop
  useEffect(() => {
    let timer: number | null = null;
    if (isPlaying && dashboardTab === 'playback') {
      const interval = 5000 / playbackSpeed;
      timer = window.setInterval(() => {
        setPlaybackIndex(prev => {
          if (prev >= replayRecords.length - 1) {
            window.setTimeout(() => setIsPlaying(false), 0);
            return prev;
          }
          return prev + 1;
        });
      }, interval);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isPlaying, playbackSpeed, dashboardTab, replayRecords.length]);

  const loadRecordedReplay = async (icao24: string) => {
    setIsPlaying(false);
    setReplayRecords([]);
    setReplayMessage('Loading stored telemetry…');
    try {
      const response = await fetch(`${API_BASE}/api/v1/aircraft/${encodeURIComponent(icao24)}/replay`);
      if (!response.ok) throw new Error(`History request failed (${response.status})`);
      const rows = await response.json() as RecordedState[];
      setReplayRecords(rows);
      setPlaybackIndex(0);
      setReplayMessage(rows.length ? `${rows.length} recorded observations loaded for ${icao24.toUpperCase()}.` : 'No stored observations exist for this aircraft yet.');
    } catch (error) {
      setReplayMessage(error instanceof Error ? error.message : 'Unable to load recorded history.');
    }
  };

  // Tick clock
  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setSimulatedTime(now.toTimeString().split(' ')[0] + ' UTC');
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  // Save config settings
  const handleSaveConfig = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      if (res.ok) {
        alert("Config parameters saved successfully!");
      }
    } catch (err) {
      alert("Failed to save config.");
    }
  };

  // Reset to default thresholds
  const handleResetConfig = () => {
    setConfig({
      max_implied_speed_kmh: 1200.0,
      duplicate_icao_dist_km: 50.0,
      max_vertical_rate_ms: 50.0,
      max_ground_altitude_m: 100.0,
      max_ground_speed_ms: 77.0,
      min_flight_speed_ms: 20.0
    });
  };

  // POST /api/v1/model-runs/replay trigger
  const handleReplaySession = async () => {
    setIsReplaying(true);
    try {
      // Save current configs first
      await fetch(`${API_BASE}/api/v1/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      
      const res = await fetch(`${API_BASE}/api/v1/model-runs/replay`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setReplayResult(data);
      } else {
        alert("Replay validation failed.");
      }
    } catch (err) {
      alert("Error connecting to replay validator service.");
    }
    setIsReplaying(false);
  };

  // Select active flight array based on current tab view
  const activeFlights = useMemo((): Flight[] => {
    if (dashboardTab === 'playback') {
      const row = replayRecords[playbackIndex];
      if (!row) return [];
      return [{
        id: row.icao24,
        callsign: row.callsign?.trim() || row.icao24.toUpperCase(),
        squawk: row.squawk ?? undefined,
        altitude: Math.round(row.altitude_m * 3.28084),
        speed: Math.round(row.velocity_ms * 1.94384),
        heading: Math.round(row.heading_deg),
        status: 'unknown',
        lat: row.latitude,
        lng: row.longitude,
        history: []
      }];
    }
    return flights;
  }, [flights, playbackIndex, dashboardTab, replayRecords]);

  // Filter and Sort active flight list
  const filteredFlights = useMemo(() => {
    return activeFlights.filter(f => {
      if (activeFilter === 'all') return true;
      return f.status === activeFilter;
    });
  }, [activeFlights, activeFilter]);

  // Sort: Flagged-first
  const sortedFlights = useMemo(() => {
    return [...filteredFlights].sort((a, b) => {
      const severityMap = { 'critical': 3, 'suspicious': 2, 'normal': 1, 'unknown': 0 };
      if (severityMap[a.status] !== severityMap[b.status]) {
        return severityMap[b.status] - severityMap[a.status];
      }
      return (a.trustScore ?? 0) - (b.trustScore ?? 0);
    });
  }, [filteredFlights]);

  const selectedFlight = activeFlights.find(f => f.id === selectedFlightId);

  // Stats Card Calculations
  const stats = useMemo(() => {
    const total = activeFlights.length;
    const anomalies = activeFlights.filter(f => f.status === 'suspicious' || f.status === 'critical').length;
    const scoredFlights = activeFlights.filter((flight) => flight.trustScore != null);
    const avgTrust = scoredFlights.length > 0
      ? Math.round(scoredFlights.reduce((acc, f) => acc + (f.trustScore ?? 0), 0) / scoredFlights.length * 10) / 10
      : null;
    return { total, anomalies, avgTrust };
  }, [activeFlights]);

  // Sort and Paginate Alerts
  const sortedAlerts = useMemo(() => {
    return [...alerts].sort((a, b) => {
      const valA = a[sortField];
      const valB = b[sortField];
      if (typeof valA === 'string') {
        return sortAsc ? valA.localeCompare(valB as string) : (valB as string).localeCompare(valA);
      }
      return sortAsc ? (valA as number) - (valB as number) : (valB as number) - (valA as number);
    });
  }, [alerts, sortField, sortAsc]);

  const paginatedAlerts = useMemo(() => {
    const start = alertPage * alertsPerPage;
    return sortedAlerts.slice(start, start + alertsPerPage);
  }, [sortedAlerts, alertPage]);

  // Client-Side CSV Export
  const exportAlertsCSV = () => {
    const headers = ["ID", "Timestamp", "Callsign", "ICAO24", "Type/Reason", "Severity", "Impact", "Acknowledged"];
    const rows = sortedAlerts.map(a => [
      a.id, a.timestamp, a.callsign, a.icao24, a.type, a.severity, a.scoreImpact, a.acknowledged
    ]);
    const csvContent = "data:text/csv;charset=utf-8," 
      + [headers.join(","), ...rows.map(e => e.map(val => `"${val}"`).join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `airguard_alerts_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const downloadSessionReportPDF = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/reports/session`);
      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', `airguard_session_report_${new Date().toISOString().split('T')[0]}.pdf`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      } else {
        alert("Failed to compile session report.");
      }
    } catch (err) {
      alert("Connection to backend report engine failed.");
    }
  };

  // Trigger server-side alert acknowledgment
  const handleAcknowledge = async (id: string) => {
    acknowledgeAlert(id);
    try {
      await fetch(`${API_BASE}/api/v1/alerts/${id}/acknowledge`, { method: 'POST' });
    } catch (e) {
      // Degrade gracefully if offline
    }
  };

  const createGeofence = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const payload = {
      name: geofenceForm.name.trim(),
      latitude: Number(geofenceForm.latitude),
      longitude: Number(geofenceForm.longitude),
      radius_km: Number(geofenceForm.radius_km),
      enabled: true
    };
    if (!payload.name || !Number.isFinite(payload.latitude) || !Number.isFinite(payload.longitude) || !Number.isFinite(payload.radius_km)) {
      setGeofenceMessage('Enter a name and valid numeric coordinates/radius.');
      return;
    }
    try {
      const response = await fetch(`${API_BASE}/api/v1/geofences`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
      });
      if (!response.ok) throw new Error(`Could not save geofence (${response.status}).`);
      const created = await response.json() as GeofenceRecord;
      setGeofences((current) => [...current, created]);
      setGeofenceForm({ name: '', latitude: '', longitude: '', radius_km: '5' });
      setGeofenceMessage('Zone saved. Aircraft entering it on a subsequent live report will be flagged.');
    } catch (error) {
      setGeofenceMessage(error instanceof Error ? error.message : 'Geofence service is unavailable.');
    }
  };

  const removeGeofence = async (id: number) => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/geofences/${id}`, { method: 'DELETE' });
      if (!response.ok) throw new Error(`Could not delete geofence (${response.status}).`);
      setGeofences((current) => current.filter((zone) => zone.id !== id));
    } catch (error) {
      setGeofenceMessage(error instanceof Error ? error.message : 'Could not remove geofence.');
    }
  };

  const updateIncident = async (incident: IncidentRecord, status: string, comment?: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/incidents/${encodeURIComponent(incident.id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status, comment: comment || null })
      });
      if (!response.ok) throw new Error(`Incident update failed (${response.status})`);
      const updated = await response.json() as IncidentRecord;
      setIncidentRecords((current) => current.map((item) => item.id === updated.id ? updated : item));
    } catch (error) {
      alert(error instanceof Error ? error.message : 'Unable to update incident.');
    }
  };

  // Virtualized row renderer
  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => {
    const flight = sortedFlights[index];
    if (!flight) return null;
    const isSelected = flight.id === selectedFlightId;
    const statusColor = 
      flight.status === 'critical' ? 'border-rose-500 text-rose-400' : 
      flight.status === 'suspicious' ? 'border-amber-500 text-amber-400' : 
      flight.status === 'unknown' ? 'border-slate-600 text-slate-400' :
      'border-emerald-500 text-emerald-400';
    
    const statusText = 
      flight.status === 'critical' ? '▲ [CRIT]' : 
      flight.status === 'suspicious' ? '◆ [WARN]' : 
      flight.status === 'unknown' ? '— [NO SCORE]' :
      '● [OK]';

    const handleKeyDown = (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        setSelectedFlightId(flight.id);
      }
    };
    
    return (
      <div style={style} className="px-2">
        <div 
          role="button"
          tabIndex={0}
          aria-label={`Flight ${flight.callsign || 'unknown'}, status ${flight.status}, ${flight.trustScore == null ? 'risk score unavailable' : `trust score ${flight.trustScore} percent`}`}
          onClick={() => setSelectedFlightId(flight.id)}
          onKeyDown={handleKeyDown}
          className={`p-2.5 rounded border cursor-pointer transition-all duration-150 focus:outline-none focus:ring-1 focus:ring-cyan-500 ${
            isSelected 
              ? 'bg-slate-800 border-cyan-500 shadow-lg shadow-cyan-500/5' 
              : 'bg-[#0b1324] border-cyan-950/30 hover:bg-slate-800/40'
          }`}
        >
          <div className="flex justify-between items-center mb-1">
            <span className="font-bold text-xs tracking-wide text-slate-100">{flight.callsign}</span>
            <span className={`text-[9px] px-2 py-0.5 rounded-full border font-mono ${statusColor}`}>
              {statusText} {flight.trustScore == null ? 'Risk unavailable' : `${flight.trustScore}% Trust`}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-x-2 text-[9px] text-slate-500 font-mono">
            <div>Alt: {flight.altitude.toLocaleString()} ft</div>
            <div>Spd: {flight.speed} kts</div>
          </div>
        </div>
      </div>
    );
  };

  // Analytics are derived only from actual stored replay runs and received alerts.
  const accuracyData = [...evaluationRuns].reverse().filter((run) => run.notes.startsWith('Positive-only labeled replay:') && run.recall != null).map((run) => ({
    time: new Date(run.run_at).toLocaleDateString(), recall: run.recall as number
  }));
  const typeData = Object.entries(alerts.reduce<Record<string, number>>((counts, alert) => {
    const label = alert.ruleFlags?.[0] || alert.type.split(/[;:]/)[0]?.slice(0, 24) || 'Recorded alert';
    counts[label] = (counts[label] ?? 0) + 1;
    return counts;
  }, {})).map(([type, count]) => ({ type, count }));
  const volumeData = [{ time: 'NOW', volume: flights.length }];

  return (
    <div className="min-h-screen bg-[#05080f] text-slate-400 flex flex-col font-mono relative overflow-hidden">
      {/* Grid overlay background */}
      <div className="absolute inset-0 grid-overlay pointer-events-none z-0"></div>

      {/* --- Top Header Navigation --- */}
      <header className="app-header relative z-10 border-b border-cyan-950/40 bg-[#060b14]/90 px-6 py-4 flex items-center justify-between backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded bg-gradient-to-tr from-cyan-600 to-blue-600 flex items-center justify-center font-bold text-xl tracking-wider text-black shadow-lg shadow-cyan-500/10">
            AG
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-slate-100 m-0 leading-none">
              AIRGUARD
            </h1>
            <span className="text-[9px] text-cyan-500 font-semibold tracking-widest uppercase">ADS-B trust verification</span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav aria-label="Main navigation" className="app-primary-nav flex bg-[#09101d] border border-cyan-950/60 rounded p-0.5">
          <button 
            onClick={() => setActiveView('about')}
            className={`text-xs font-semibold px-4 py-1.5 rounded transition-all ${
              activeView === 'about' 
                ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20' 
                : 'text-slate-500 hover:text-slate-300 border border-transparent'
            }`}
          >
            SYSTEM OVERVIEW
          </button>
          <button 
            onClick={() => {
              setActiveView('dashboard');
              setDashboardTab('radar');
            }}
            className={`text-xs font-semibold px-4 py-1.5 rounded transition-all ${
              activeView === 'dashboard' && dashboardTab !== 'playback' && dashboardTab !== 'analytics'
                ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20' 
                : 'text-slate-500 hover:text-slate-300 border border-transparent'
            }`}
          >
            TACTICAL SCREEN
          </button>
          <button 
            onClick={() => {
              setActiveView('dashboard');
              setDashboardTab('playback');
              setSelectedFlightId(null);
            }}
            className={`text-xs font-semibold px-4 py-1.5 rounded transition-all ${
              activeView === 'dashboard' && dashboardTab === 'playback'
                ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20' 
                : 'text-slate-500 hover:text-slate-300 border border-transparent'
            }`}
          >
            HISTORICAL PLAYBACK
          </button>
          <button 
            onClick={() => {
              setActiveView('dashboard');
              setDashboardTab('analytics');
              setSelectedFlightId(null);
            }}
            className={`text-xs font-semibold px-4 py-1.5 rounded transition-all ${
              activeView === 'dashboard' && dashboardTab === 'analytics'
                ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20' 
                : 'text-slate-500 hover:text-slate-300 border border-transparent'
            }`}
          >
            ANALYTICS & CONFIG
          </button>
        </nav>

        {/* Live system state bar */}
        <div className="flex items-center gap-4">
          <div className="hidden md:flex items-center gap-2 px-3 py-1 rounded bg-[#09101d] border border-slate-800">
            <span className="text-[10px] text-slate-500">UTC:</span>
            <span className="text-[10px] text-slate-300">{simulatedTime}</span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-[#09101d] border border-slate-900">
            <span className="relative flex h-2 w-2">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                websocketStatus === 'connected' ? 'bg-emerald-400' : 
                websocketStatus === 'connecting' ? 'bg-amber-400 animate-pulse' : 'bg-rose-400'
              }`}></span>
              <span className={`relative inline-flex rounded-full h-2 w-2 ${
                websocketStatus === 'connected' ? 'bg-emerald-500' : 
                websocketStatus === 'connecting' ? 'bg-amber-500' : 'bg-rose-500'
              }`}></span>
            </span>
            <span className="text-[9px] text-slate-300 font-bold tracking-wider uppercase">
              STREAM: {websocketStatus}
            </span>
          </div>
        </div>
      </header>

      {/* --- Landing / About View --- */}
      {activeView === 'about' && (
        <div className="relative z-10 flex-1 flex flex-col justify-between py-12 px-6 max-w-6xl mx-auto w-full">
          
          {/* Hero Section */}
          <section className="mt-8 text-center max-w-4xl mx-auto">
            <span className="text-xs text-cyan-500 tracking-widest uppercase font-bold border border-cyan-950 px-3 py-1.5 rounded bg-cyan-950/10">
              TRUST SCORING RADAR
            </span>
            <h2 className="text-3xl md:text-5xl font-black mt-8 text-slate-100 leading-tight">
              FlightRadar24 tells you where a plane is.<br />
              <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                We tell you whether you should trust that.
              </span>
            </h2>
            <p className="mt-6 text-sm md:text-base text-slate-400 max-w-2xl mx-auto leading-relaxed">
              Standard civil aviation transponders broadcast position data unauthenticated. AirGuard intercepts mode S feeds, analyzes signal geometry, and runs deep autoencoders to flag spoofed trajectories in real-time.
            </p>
            <div className="mt-8 flex justify-center gap-4">
              <button 
                onClick={() => {
                  setActiveView('dashboard');
                  setDashboardTab('radar');
                }}
                className="bg-cyan-600 hover:bg-cyan-500 text-black font-bold text-xs py-3 px-8 rounded transition-all shadow-lg shadow-cyan-500/25 hover:-translate-y-0.5"
              >
                OPEN RADAR CONSOLE
              </button>
              <a 
                href="#how-it-works"
                className="border border-cyan-950 bg-slate-950/40 hover:bg-cyan-950/10 text-slate-300 font-bold text-xs py-3 px-8 rounded transition-all flex items-center"
              >
                SYSTEM DETAILS
              </a>
            </div>
          </section>

          {/* Live Mini Stat Strip */}
          <section className="my-16 grid grid-cols-2 md:grid-cols-4 gap-4 border border-cyan-950/40 bg-[#060b14]/60 p-4 rounded backdrop-blur-sm max-w-4xl mx-auto w-full text-xs font-mono">
            <div className="p-3 border-r border-cyan-950/40 last:border-none">
              <span className="text-slate-500 block mb-1">CIRCUIT BREAKER</span>
              <span className="text-emerald-400 font-bold tracking-wider">{healthData.circuit_breaker_state}</span>
            </div>
            <div className="p-3 border-r border-cyan-950/40 last:border-none">
              <span className="text-slate-500 block mb-1">QUEUE DEPTH</span>
              <span className="text-cyan-400 font-bold">{healthData.queue_depth} vectors</span>
            </div>
            <div className="p-3 border-r border-cyan-950/40 last:border-none">
              <span className="text-slate-500 block mb-1">POLL LATENCY</span>
              <span className="text-cyan-400 font-bold">{healthData.poll_latency_ms.toFixed(1)} ms</span>
            </div>
            <div className="p-3 last:border-none">
              <span className="text-slate-300 font-bold">
                {healthData.last_successful_poll ? healthData.last_successful_poll.split('T')[1]?.substring(0, 8) || 'ONLINE' : 'ACTIVE'}
              </span>
            </div>
          </section>

          {/* Problem Statement Section */}
          <section className="border-t border-cyan-950/40 pt-16 grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
            <div>
              <h3 className="text-xl font-bold text-slate-200 uppercase mb-4 tracking-wider">The Vulnerability in the Sky</h3>
              <p className="text-xs md:text-sm leading-relaxed mb-4">
                ADS-B signals are completely unencrypted. Any hobbyist with a transmitter can broadcast false coordinates, creating &quot;ghost aircraft&quot; or altering the reported routes of real planes.
              </p>
              <p className="text-xs md:text-sm leading-relaxed">
                AirGuard solves this by decoupling trust from the transponder reports. We reconstruct target physics and score consistency statically and dynamically.
              </p>
            </div>
            <div className="bg-[#060b14]/40 border border-cyan-950/40 rounded p-4 font-mono text-[11px] leading-relaxed shadow-lg">
              <span className="text-cyan-500 block mb-2 font-bold font-mono">REAL-TIME DATA FLOW INCIDENT LOG</span>
              <div className="space-y-1.5 max-h-[140px] overflow-y-auto text-slate-500">
                Live incident entries appear here when telemetry from a connected source produces an alert.
              </div>
            </div>
          </section>

          {/* How It Works Flow (4 Steps) */}
          <section id="how-it-works" className="mt-20 border-t border-cyan-950/40 pt-16">
            <h3 className="text-center text-xl font-bold tracking-widest text-slate-100 uppercase mb-12">HOW IT WORKS</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              {/* Step 1 */}
              <div className="p-5 border border-cyan-950/40 bg-[#060b14]/30 rounded flex flex-col justify-between h-[240px]">
                <div>
                  <span className="text-cyan-500 font-bold text-xs tracking-widest">01 / RECEIVE</span>
                  <h4 className="text-sm font-bold text-slate-200 mt-2">SDR Ingestion</h4>
                  <p className="text-[11px] text-slate-500 mt-2 leading-relaxed">Intercepts direct RF Mode S frames from local dump1090 receivers and caches timing.</p>
                </div>
                {/* Visual */}
                <div className="flex gap-1 items-end h-8 pb-1">
                  <div className="bg-cyan-500/30 h-2 w-1.5"></div>
                  <div className="bg-cyan-500/50 h-4 w-1.5"></div>
                  <div className="bg-cyan-500 h-7 w-1.5 animate-pulse"></div>
                  <div className="bg-cyan-500/60 h-3 w-1.5"></div>
                  <div className="bg-cyan-500/20 h-1 w-1.5"></div>
                </div>
              </div>

              {/* Step 2 */}
              <div className="p-5 border border-cyan-950/40 bg-[#060b14]/30 rounded flex flex-col justify-between h-[240px]">
                <div>
                  <span className="text-cyan-500 font-bold text-xs tracking-widest">02 / DECODE</span>
                  <h4 className="text-sm font-bold text-slate-200 mt-2">Mode S Extraction</h4>
                  <p className="text-[11px] text-slate-500 mt-2 leading-relaxed">Extracts latitude, longitude, squawks, and signal metrics, padding missing attributes.</p>
                </div>
                {/* Visual */}
                <div className="font-mono text-[9px] text-cyan-500/70 border border-cyan-950 p-1.5 rounded bg-black/40 overflow-hidden">
                  <span>Raw transponder fields are shown only when supplied by the connected receiver.</span>
                </div>
              </div>

              {/* Step 3 */}
              <div className="p-5 border border-cyan-950/40 bg-[#060b14]/30 rounded flex flex-col justify-between h-[240px]">
                <div>
                  <span className="text-cyan-500 font-bold text-xs tracking-widest">03 / DETECT</span>
                  <h4 className="text-sm font-bold text-slate-200 mt-2">Aerodynamic Verification</h4>
                  <p className="text-[11px] text-slate-500 mt-2 leading-relaxed">Checks climb envelope limits, Haversine position deltas, and alt-velocity correlations.</p>
                </div>
                {/* Visual */}
                <div className="relative h-12 w-full border border-cyan-950/60 rounded overflow-hidden flex items-center justify-center">
                  <div className="absolute inset-0 bg-[#06b6d4]/5 radar-sweep-effect"></div>
                  <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 absolute"></div>
                </div>
              </div>

              {/* Step 4 */}
              <div className="p-5 border border-cyan-950/40 bg-[#060b14]/30 rounded flex flex-col justify-between h-[240px]">
                <div>
                  <span className="text-cyan-500 font-bold text-xs tracking-widest">04 / VERIFY</span>
                  <h4 className="text-sm font-bold text-slate-200 mt-2">Ensemble Trust Scoring</h4>
                  <p className="text-[11px] text-slate-500 mt-2 leading-relaxed">Feeds RF+GB ML classifiers and PyTorch Autoencoders to generate a unified risk rating.</p>
                </div>
                {/* Visual */}
                <div className="grid grid-cols-3 gap-1.5 text-center text-[9px]">
                  <div className="col-span-3 p-1 border border-cyan-950 bg-cyan-950/20 text-slate-400 rounded">Model scores are populated from the active backend analysis.</div>
                </div>
              </div>
            </div>
          </section>

        </div>
      )}

      {/* --- Tactical Dashboard View --- */}
      {activeView === 'dashboard' && (
        <div className="app-dashboard flex-1 flex flex-col min-h-0 relative z-10 px-6 py-4">
          
          {/* Sub-Header Tabs */}
          <div className="flex border-b border-cyan-950/30 mb-4 gap-2 text-xs font-bold items-center justify-between">
            <div className="flex gap-2">
              <button
                onClick={() => setDashboardTab('radar')}
                className={`pb-2 px-3 border-b-2 transition-all ${
                  dashboardTab === 'radar' 
                    ? 'border-cyan-400 text-cyan-400' 
                    : 'border-transparent text-slate-500 hover:text-slate-300'
                }`}
              >
                RADAR OPERATIONS
              </button>
              <button
                onClick={() => setDashboardTab('alerts')}
                className={`pb-2 px-3 border-b-2 transition-all ${
                  dashboardTab === 'alerts' 
                    ? 'border-cyan-400 text-cyan-400' 
                    : 'border-transparent text-slate-500 hover:text-slate-300'
                }`}
              >
                ALERTS AUDIT LOG
              </button>
              <button
                onClick={() => setDashboardTab('playback')}
                className={`pb-2 px-3 border-b-2 transition-all ${
                  dashboardTab === 'playback' 
                    ? 'border-cyan-400 text-cyan-400' 
                    : 'border-transparent text-slate-500 hover:text-slate-300'
                }`}
              >
                HISTORICAL PLAYBACK
              </button>
              <button
                onClick={() => setDashboardTab('analytics')}
                className={`pb-2 px-3 border-b-2 transition-all ${
                  dashboardTab === 'analytics' 
                    ? 'border-cyan-400 text-cyan-400' 
                    : 'border-transparent text-slate-500 hover:text-slate-300'
                }`}
              >
                ANALYTICS & CONFIG
              </button>
            </div>
          </div>

          {/* Tab 1 & Tab 3: Radar Screen Operations */}
          {(dashboardTab === 'radar' || dashboardTab === 'playback') && (
            <div className="flex-1 flex flex-col min-h-0">
              
              {/* Playback Control Panel */}
              {dashboardTab === 'playback' && (
                <div className="mb-4 p-4 bg-[#070d18] border border-cyan-950/50 rounded shadow-md font-mono text-xs flex flex-col md:flex-row items-center justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <select
                      value={selectedFlightId ?? ''}
                      onChange={(event) => {
                        if (event.target.value) {
                          setSelectedFlightId(event.target.value);
                          void loadRecordedReplay(event.target.value);
                        }
                      }}
                      className="bg-slate-950 border border-cyan-950/50 rounded px-3 py-2 text-slate-200"
                    >
                      <option value="">Choose recorded aircraft</option>
                      {flights.map((flight) => <option key={flight.id} value={flight.id}>{flight.callsign} · {flight.id.toUpperCase()}</option>)}
                    </select>
                    <button
                      onClick={() => replayRecords.length > 0 && setIsPlaying(!isPlaying)}
                      disabled={replayRecords.length === 0}
                      className={`px-4 py-2 rounded text-black font-bold font-mono transition-all text-[11px] ${
                        isPlaying ? 'bg-amber-500 hover:bg-amber-400' : 'bg-cyan-500 hover:bg-cyan-400'
                      }`}
                    >
                      {isPlaying ? 'PAUSE PLAYBACK' : 'START PLAYBACK'}
                    </button>

                    <div className="flex bg-slate-950 border border-cyan-950/30 rounded p-0.5">
                      {([1, 5, 20] as const).map(speed => (
                        <button
                          key={speed}
                          onClick={() => setPlaybackSpeed(speed)}
                          className={`px-3 py-1 rounded text-[10px] transition-all font-bold ${
                            playbackSpeed === speed 
                              ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20' 
                              : 'text-slate-500 hover:text-slate-300 border border-transparent'
                          }`}
                        >
                          {speed}x
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="flex-1 flex items-center gap-4 w-full">
                    <span className="text-[10px] text-slate-500">TIMELINE:</span>
                    <input
                      type="range"
                      min={0}
                      max={Math.max(0, replayRecords.length - 1)}
                      value={playbackIndex}
                      disabled={replayRecords.length === 0}
                      onChange={(e) => setPlaybackIndex(parseInt(e.target.value, 10))}
                      className="flex-1 accent-cyan-500 bg-slate-950 h-1 rounded cursor-pointer border border-cyan-950/30"
                    />
                    <span className="text-cyan-400 font-bold text-[10px] min-w-[140px] text-right">
                      {replayRecords[playbackIndex]?.received_at ? new Date(replayRecords[playbackIndex].received_at).toISOString() : 'NO RECORDED TIME'}
                    </span>
                  </div>

                  <div className="w-full md:w-[320px] bg-slate-950 border border-cyan-950/40 p-2 rounded max-h-[44px] overflow-hidden text-[9px] leading-relaxed text-cyan-500/90 font-mono shadow-inner">
                    <span className="text-slate-500 font-bold block">RECORDED DATA</span>
                    <span className="truncate block">{replayMessage} {replayRecords[playbackIndex] ? `· ${replayRecords[playbackIndex].source}` : ''}</span>
                  </div>
                </div>
              )}

              {/* Main Workspace Frame */}
              <div className="flex-1 grid grid-cols-12 gap-6 min-h-0">
                {/* Left 65%: Cesium Globe */}
                <section className="col-span-8 bg-[#090f1d]/50 border border-cyan-950/50 rounded shadow-xl overflow-hidden relative flex flex-col">
                  <div className="absolute top-4 left-4 z-10 flex items-center gap-2 bg-slate-950/85 border border-cyan-950/60 rounded px-3 py-1.5 backdrop-blur-md">
                    <div className="w-2.5 h-2.5 rounded-full bg-cyan-500 animate-pulse"></div>
                    <span className="text-[9px] font-bold tracking-widest text-slate-300">
                      {dashboardTab === 'playback' ? "HISTORICAL RECONSTRUCTION" : "CESIUM 3D GLOBE OVERLAY"}
                    </span>
                  </div>
                  
                  <div className="flex-1 w-full relative" aria-label="3D Cesium map visualizing tracked targets" role="application">
                    <CesiumErrorBoundary>
                      <Viewer full className="w-full h-full">
                        {dashboardTab === 'radar' && geofences.filter((zone) => zone.enabled).map((zone) => (
                          <Entity
                            key={`geofence-${zone.id}`}
                            position={Cartesian3.fromDegrees(zone.longitude, zone.latitude, 0)}
                            name={`${zone.name} · ${zone.radius_km} km`}
                          >
                            <EllipseGraphics
                              semiMajorAxis={zone.radius_km * 1000}
                              semiMinorAxis={zone.radius_km * 1000}
                              material={Color.ORANGE.withAlpha(0.12)}
                              outline
                              outlineColor={Color.ORANGE.withAlpha(0.8)}
                              outlineWidth={2}
                            />
                          </Entity>
                        ))}
                        {activeFlights.map(f => {
                          if (typeof f.lng !== 'number' || typeof f.lat !== 'number' || isNaN(f.lng) || isNaN(f.lat)) {
                            return null;
                          }
                          const position = Cartesian3.fromDegrees(f.lng, f.lat, f.altitude * 0.3048);
                          const color = 
                            f.status === 'critical' ? Color.RED :
                            f.status === 'suspicious' ? Color.ORANGE :
                            f.status === 'unknown' ? Color.GRAY :
                            Color.GREEN;
                            
                          const trailPositions = [
                            Cartesian3.fromDegrees(f.lng, f.lat, f.altitude * 0.3048),
                            ...(f.history || [])
                              .filter(h => typeof h.lng === 'number' && typeof h.lat === 'number' && !isNaN(h.lng) && !isNaN(h.lat))
                              .map(h => Cartesian3.fromDegrees(h.lng, h.lat, f.altitude * 0.3048))
                          ];
                          
                          return (
                            <Entity 
                              key={f.id} 
                              position={position}
                              name={f.callsign}
                              onClick={() => setSelectedFlightId(f.id)}
                            >
                              <PointGraphics pixelSize={8} color={color} outlineColor={Color.BLACK} outlineWidth={1.5} />
                              {trailPositions.length > 1 && (
                                <PolylineGraphics
                                  positions={trailPositions}
                                  width={1.5}
                                  material={color.withAlpha(0.4)}
                                />
                              )}
                            </Entity>
                          );
                        })}
                      </Viewer>
                    </CesiumErrorBoundary>
                  </div>
                </section>

                {/* Right 35%: Drawer / Aircraft Detail drawer */}
                <section className="col-span-4 bg-[#090f1d]/50 border border-cyan-950/50 rounded shadow-xl flex flex-col overflow-hidden backdrop-blur-md">
                  
                  {!selectedFlightId ? (
                    <div className="flex flex-col h-full">
                      <div className="px-4 py-3 border-b border-cyan-950/40 flex items-center justify-between">
                        <div>
                          <h2 className="text-xs font-bold tracking-wider text-slate-400 uppercase m-0">TACTICAL TARGETS</h2>
                          <span className="text-[8px] text-slate-500 font-mono">Flagged-First / Virtualized (500+ limit)</span>
                        </div>
                        <div className="flex bg-slate-950 border border-slate-900 rounded p-0.5">
                          {(['all', 'suspicious', 'critical'] as const).map(filterType => (
                            <button
                              key={filterType}
                              onClick={() => setActiveFilter(filterType)}
                              className={`text-[9px] font-mono px-2 py-1 rounded transition-all capitalize ${
                                activeFilter === filterType 
                                  ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-bold' 
                                  : 'text-slate-500 hover:text-slate-300 border border-transparent'
                              }`}
                            >
                              {filterType}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="flex-1 min-h-0 py-2">
                        <List
                          height={540}
                          itemCount={sortedFlights.length}
                          itemSize={76}
                          width="100%"
                        >
                          {Row}
                        </List>
                      </div>
                    </div>
                  ) : (
                    // HIGHLY POLISHED DETAIL DRAWER OVERLAY
                    <div className="flex flex-col h-full bg-[#070c17]/95 border-l border-cyan-950/50 p-5 overflow-y-auto">
                      <div className="flex justify-between items-center border-b border-cyan-950/40 pb-3 mb-4">
                        <div>
                          <span className="text-[9px] font-bold text-cyan-500 tracking-widest uppercase">Target Details</span>
                          <h3 className="text-lg font-bold text-slate-100 leading-none mt-1">{selectedFlight?.callsign}</h3>
                        </div>
                        <button 
                          onClick={() => setSelectedFlightId(null)}
                          className="text-[10px] bg-slate-950 hover:bg-cyan-950/30 border border-cyan-950 text-slate-400 hover:text-cyan-400 px-3 py-1 rounded"
                        >
                          CLOSE DRAWER
                        </button>
                      </div>

                      {selectedFlight && (
                        <div className="space-y-6 text-xs font-mono">
                          
                          <div className="grid grid-cols-2 gap-4 bg-slate-950/60 p-3 rounded border border-cyan-950/20">
                            <div>
                              <span className="text-slate-500 text-[8px] block">ICAO ADDRESS</span>
                              <span className="text-slate-200 font-bold">{selectedFlight.id.toUpperCase()}</span>
                            </div>
                            <div>
                              <span className="text-slate-500 text-[8px] block">SQUAWK CODE</span>
                              <span className="text-slate-200 font-bold">{selectedFlight.squawk || 'Not reported by source'}</span>
                            </div>
                            <div>
                              <span className="text-slate-500 text-[8px] block">ALTITUDE (MSL)</span>
                              <span className="text-slate-200 font-bold">{selectedFlight.altitude.toLocaleString()} ft</span>
                            </div>
                            <div>
                              <span className="text-slate-500 text-[8px] block">GROUND SPEED</span>
                              <span className="text-slate-200 font-bold">{selectedFlight.speed} kts</span>
                            </div>
                            <div>
                              <span className="text-slate-500 text-[8px] block">HEADING</span>
                              <span className="text-slate-200 font-bold">{selectedFlight.heading}°</span>
                            </div>
                            <div>
                              <span className="text-slate-500 text-[8px] block">SIGNAL STRENGTH</span>
                              <span className="text-slate-200 font-bold">{selectedFlight.signalStrength ?? 'Not reported by source'}{selectedFlight.signalStrength != null ? ' dBm' : ''}</span>
                            </div>
                          </div>

                          <div className="p-3 rounded border border-cyan-950/30 bg-slate-950/50 space-y-2">
                            <span className="text-[10px] text-slate-400 font-bold block">LIVE EMERGENCY ASSESSMENT</span>
                            <div className={selectedFlight.emergencyCode ? 'text-rose-400' : 'text-slate-400'}>
                              {selectedFlight.emergencyCode ? `Transponder emergency code ${selectedFlight.emergencyCode}` : 'No emergency squawk currently reported'}
                            </div>
                            <div className={selectedFlight.rapidDescent ? 'text-rose-400' : 'text-slate-400'}>
                              {selectedFlight.rapidDescent ? 'Rapid descent detected from live vertical-rate/altitude data' : 'No rapid descent detected in available telemetry'}
                            </div>
                            <div className={selectedFlight.routeDeviationNm != null ? 'text-amber-300' : 'text-slate-500'}>
                              {selectedFlight.routeDeviationNm != null ? `Distance from supplied filed route: ${selectedFlight.routeDeviationNm.toFixed(1)} NM` : 'Filed route unavailable from current telemetry source'}
                            </div>
                            <div className="text-slate-500">Diversion airports: unavailable until an operational airport and runway data source is connected.</div>
                          </div>

                          <div className="p-3 rounded border border-cyan-950/30 bg-slate-950/50">
                            <span className="text-[10px] text-slate-400 font-bold block mb-2">RECORDED INCIDENT TIMELINE</span>
                            {incidentRecords.filter((incident) => incident.icao24 === selectedFlight.id).length === 0 ? (
                              <span className="text-[10px] text-slate-500">No incident events recorded for this aircraft.</span>
                            ) : (
                              <div className="space-y-4">
                                {incidentRecords.filter((incident) => incident.icao24 === selectedFlight.id).map((incident) => (
                                  <div key={incident.id} className="border-t border-cyan-950/30 pt-2">
                                    <div className="flex items-center justify-between gap-2 mb-2">
                                      <span className="text-[9px] text-amber-300">{incident.id} · {incident.reason}</span>
                                      <select value={incident.status} onChange={(event) => void updateIncident(incident, event.target.value)} className="bg-slate-950 border border-cyan-950/50 rounded px-2 py-1 text-[9px] text-slate-200">
                                        {['new', 'investigating', 'acknowledged', 'resolved', 'archived'].map((status) => <option key={status} value={status}>{status.toUpperCase()}</option>)}
                                      </select>
                                    </div>
                                    <ol className="space-y-2">
                                      {(incident.timeline ?? []).map((event, index) => (
                                        <li key={`${event.at}-${index}`} className="text-[10px] text-slate-300 border-l border-cyan-800 pl-2">
                                          <time className="text-cyan-500">{new Date(event.at).toLocaleString()}</time> · {event.signals.join(', ') || event.type}
                                        </li>
                                      ))}
                                    </ol>
                                    {(incident.comments ?? []).map((comment, index) => <p key={`${comment.at}-${index}`} className="text-[10px] text-slate-400 mt-1">Operator note · {comment.text}</p>)}
                                    <form className="flex gap-2 mt-2" onSubmit={(event) => {
                                      event.preventDefault();
                                      const form = event.currentTarget;
                                      const input = form.elements.namedItem('incident-comment') as HTMLInputElement;
                                      if (input.value.trim()) {
                                        void updateIncident(incident, incident.status, input.value.trim());
                                        input.value = '';
                                      }
                                    }}>
                                      <input name="incident-comment" placeholder="Add an operator note" className="min-w-0 flex-1 bg-slate-950 border border-cyan-950/50 rounded px-2 py-1 text-[10px] text-slate-200" />
                                      <button className="border border-cyan-900 px-2 py-1 rounded text-[9px] text-cyan-300">ADD NOTE</button>
                                    </form>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>

                          <div>
                            <span className="text-[10px] text-slate-400 font-bold block mb-2">AEROSPACE VERIFICATION RULE FLAGS</span>
                            <div className="space-y-1.5">
                              <div className="flex justify-between items-center p-2 rounded bg-slate-900 border border-cyan-950/10">
                                <span className="text-[9px]">Haversine Implied Speed</span>
                                <span className={`text-[9px] px-2 py-0.5 rounded font-bold ${
                                  selectedFlight.ruleFlags?.positionJump ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                }`}>
                                  {selectedFlight.ruleFlags?.positionJump ? "▲ [X] IMPLAUSIBLE JUMP DETECTED" : "● [✓] PASSED (Implied speed normal)"}
                                </span>
                              </div>
                              <div className="flex justify-between items-center p-2 rounded bg-slate-900 border border-cyan-950/10">
                                <span className="text-[9px]">Duplicate ICAO Detection</span>
                                <span className={`text-[9px] px-2 py-0.5 rounded font-bold ${
                                  selectedFlight.ruleFlags?.duplicateIcao ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                }`}>
                                  {selectedFlight.ruleFlags?.duplicateIcao ? "▲ [X] DUPLICATE FOUND" : "● [✓] PASSED (Unique signature)"}
                                </span>
                              </div>
                              <div className="flex justify-between items-center p-2 rounded bg-slate-900 border border-cyan-950/10">
                                <span className="text-[9px]">Vertical climb envelope limit</span>
                                <span className={`text-[9px] px-2 py-0.5 rounded font-bold ${
                                  selectedFlight.ruleFlags?.climbRate ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                }`}>
                                  {selectedFlight.ruleFlags?.climbRate ? "▲ [X] LIMIT EXCEEDED" : "● [✓] PASSED (Climb rate normal)"}
                                </span>
                              </div>
                              <div className="flex justify-between items-center p-2 rounded bg-slate-900 border border-cyan-950/10">
                                <span className="text-[9px]">Altitude-velocity correlation</span>
                                <span className={`text-[9px] px-2 py-0.5 rounded font-bold ${
                                  selectedFlight.ruleFlags?.altVelMismatch ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                }`}>
                                  {selectedFlight.ruleFlags?.altVelMismatch ? "▲ [X] PHYSICAL MISMATCH" : "● [✓] PASSED (Consistent state)"}
                                </span>
                              </div>
                            </div>
                          </div>

                          <div>
                            <span className="text-[10px] text-slate-400 font-bold block mb-1.5">GEOMETRIC MULTILATERATION FEEDBACK</span>
                            <div className="p-3 bg-slate-950 rounded border border-cyan-950/30 text-[10px]">
                              <span className="text-slate-500 block mb-1">TRILATERATION STATUS:</span>
                              <span className={`font-bold ${
                                selectedFlight.trilateration?.startsWith('Failed') ? 'text-rose-400' :
                                selectedFlight.trilateration?.startsWith('Inconclusive') ? 'text-amber-400' : 'text-emerald-400'
                              }`}>{selectedFlight.trilateration}</span>
                            </div>
                          </div>

                          <div>
                            <span className="text-[10px] text-slate-400 font-bold block mb-3">SHAP REASONING (EXPLAINABLE ML INFLUENCE)</span>
                            <div className="h-[180px] w-full bg-slate-950/60 border border-cyan-950/20 p-2 rounded flex flex-col justify-between">
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart
                                  layout="vertical"
                                  data={selectedFlight.shapValues}
                                  margin={{ top: 5, right: 10, left: 20, bottom: 5 }}
                                >
                                  <XAxis type="number" stroke="#475569" fontSize={8} />
                                  <YAxis dataKey="name" type="category" stroke="#94a3b8" fontSize={7} width={80} />
                                  <Tooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#1e293b', fontSize: '9px' }} />
                                  <Bar 
                                    dataKey="value" 
                                    fill={selectedFlight.status === 'critical' ? '#f43f5e' : selectedFlight.status === 'suspicious' ? '#f59e0b' : '#0ea5e9'} 
                                    radius={[0, 2, 2, 0]} 
                                  />
                                </BarChart>
                              </ResponsiveContainer>
                            </div>
                          </div>

                        </div>
                      )}
                    </div>
                  )}
                </section>
              </div>
            </div>
          )}

          {/* Tab 2: Full Width Alerts Audit Log Table */}
          {dashboardTab === 'alerts' && (
            <section className="flex-1 bg-[#090f1d]/50 border border-cyan-950/50 rounded shadow-xl flex flex-col overflow-hidden p-6 backdrop-blur-md">
              <div className="flex items-center justify-between border-b border-cyan-950/40 pb-4 mb-4">
                <div>
                  <h2 className="text-sm font-bold tracking-wider text-slate-300 uppercase m-0">ALERTS AUDIT LOG RECORD</h2>
                  <span className="text-[10px] text-slate-500">Security flags logged by Combined Score Rules & ML Ensemble</span>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={exportAlertsCSV}
                    className="bg-[#09101d] border border-cyan-950/60 hover:bg-cyan-950/20 text-slate-300 font-bold text-xs py-2 px-6 rounded transition-all font-mono"
                  >
                    EXPORT TO CSV
                  </button>
                  <button
                    onClick={downloadSessionReportPDF}
                    className="bg-cyan-600 hover:bg-cyan-500 text-black font-bold text-xs py-2 px-6 rounded transition-all font-mono shadow-md hover:shadow-cyan-500/25"
                  >
                    GENERATE SESSION REPORT
                  </button>
                </div>
              </div>

              <div className="flex-1 overflow-x-auto min-h-0">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-cyan-950/60 text-slate-500 font-mono uppercase text-[9px] tracking-wider">
                      <th className="py-3 px-4 cursor-pointer hover:text-slate-300" onClick={() => { setSortField('timestamp'); setSortAsc(!sortAsc); }}>
                        Timestamp {sortField === 'timestamp' && (sortAsc ? '▲' : '▼')}
                      </th>
                      <th className="py-3 px-4 cursor-pointer hover:text-slate-300" onClick={() => { setSortField('callsign'); setSortAsc(!sortAsc); }}>
                        Target Callsign {sortField === 'callsign' && (sortAsc ? '▲' : '▼')}
                      </th>
                      <th className="py-3 px-4">ICAO Address</th>
                      <th className="py-3 px-4">Anomaly Flag Details</th>
                      <th className="py-3 px-4">Severity</th>
                      <th className="py-3 px-4 cursor-pointer hover:text-slate-300" onClick={() => { setSortField('scoreImpact'); setSortAsc(!sortAsc); }}>
                        Risk Score Impact {sortField === 'scoreImpact' && (sortAsc ? '▲' : '▼')}
                      </th>
                      <th className="py-3 px-4">Acknowledge</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-cyan-950/20 font-mono text-[11px]">
                    {paginatedAlerts.length === 0 ? (
                      <tr>
                        <td colSpan={7} className="py-8 text-center text-slate-600">No alert logs available.</td>
                      </tr>
                    ) : (
                      paginatedAlerts.map(a => {
                        const sevColor = 
                          a.severity === 'high' ? 'text-rose-400' : 'text-amber-400';
                        return (
                          <tr key={a.id} onClick={() => setSelectedEvidence(a)} className={`cursor-pointer hover:bg-cyan-950/20 transition-colors ${a.acknowledged ? 'opacity-60' : ''} ${selectedEvidence?.id === a.id ? 'bg-cyan-950/20' : ''}`}>
                            <td className="py-3 px-4 text-slate-400">{a.timestamp}</td>
                            <td className="py-3 px-4 font-bold text-slate-100">{a.callsign}</td>
                            <td className="py-3 px-4 text-slate-400">{a.icao24.toUpperCase()}</td>
                            <td className="py-3 px-4 text-slate-300">{a.type}</td>
                            <td className={`py-3 px-4 uppercase font-bold ${sevColor}`}>{a.severity}</td>
                            <td className="py-3 px-4 text-rose-400">{a.scoreImpact} Trust pts</td>
                            <td className="py-3 px-4">
                              {a.acknowledged ? (
                                <span className="text-[10px] text-slate-600 font-bold border border-slate-900 bg-slate-950/40 px-2 py-1 rounded">ACKNOWLEDGED</span>
                              ) : (
                                <button
                                  onClick={() => handleAcknowledge(a.id)}
                                  className="text-[10px] bg-rose-950/20 hover:bg-rose-950/50 border border-rose-950/50 text-rose-400 px-3 py-1 rounded transition-colors"
                                >
                                  ACKNOWLEDGE
                                </button>
                              )}
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>

              {selectedEvidence && (
                <aside className="mt-4 p-4 rounded border border-cyan-800/50 bg-slate-950/80" aria-label="Selected alert evidence">
                  <div className="flex justify-between items-start gap-3 mb-3">
                    <div>
                      <h3 className="text-xs font-bold text-cyan-300 m-0">ALERT EVIDENCE · {selectedEvidence.callsign}</h3>
                      <p className="text-[10px] text-slate-400 mt-1">{selectedEvidence.type} · {new Date(selectedEvidence.timestamp).toLocaleString()}</p>
                    </div>
                    <button onClick={() => setSelectedEvidence(null)} className="text-[10px] text-slate-400 hover:text-white">CLOSE</button>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px]">
                    <div className="p-2 bg-slate-900 rounded">Risk score <b className="text-rose-300">{selectedEvidence.riskScore == null ? 'Not supplied' : `${(selectedEvidence.riskScore * 100).toFixed(1)}%`}</b></div>
                    <div className="p-2 bg-slate-900 rounded">Ensemble <b className="text-cyan-300">{selectedEvidence.modelScores?.ensemble_score == null ? '—' : selectedEvidence.modelScores.ensemble_score.toFixed(3)}</b></div>
                    <div className="p-2 bg-slate-900 rounded">Autoencoder <b className="text-cyan-300">{selectedEvidence.modelScores?.autoencoder_score == null ? '—' : selectedEvidence.modelScores.autoencoder_score.toFixed(3)}</b></div>
                    <div className="p-2 bg-slate-900 rounded">Flags <b className="text-amber-300">{selectedEvidence.ruleFlags?.length ? selectedEvidence.ruleFlags.join(', ') : 'No rule details on this event'}</b></div>
                  </div>
                  <pre className="mt-3 max-h-44 overflow-auto whitespace-pre-wrap break-words text-[9px] text-slate-300 bg-black/30 p-3 rounded">{selectedEvidence.evidence ? JSON.stringify(selectedEvidence.evidence, null, 2) : 'Stored evidence payload is unavailable for this live-stream event.'}</pre>
                </aside>
              )}

              <div className="flex items-center justify-between border-t border-cyan-950/40 pt-4 mt-4 font-mono text-[10px]">
                <span className="text-slate-500">
                  Showing {alertPage * alertsPerPage + 1} - {Math.min((alertPage + 1) * alertsPerPage, sortedAlerts.length)} of {sortedAlerts.length} logs
                </span>
                <div className="flex gap-2">
                  <button
                    disabled={alertPage === 0}
                    onClick={() => setAlertPage(alertPage - 1)}
                    className="border border-cyan-950 bg-slate-950 px-3 py-1.5 rounded text-slate-400 hover:text-cyan-400 disabled:opacity-40 disabled:pointer-events-none"
                  >
                    PREVIOUS
                  </button>
                  <button
                    disabled={(alertPage + 1) * alertsPerPage >= sortedAlerts.length}
                    onClick={() => setAlertPage(alertPage + 1)}
                    className="border border-cyan-950 bg-slate-950 px-3 py-1.5 rounded text-slate-400 hover:text-cyan-400 disabled:opacity-40 disabled:pointer-events-none"
                  >
                    NEXT
                  </button>
                </div>
              </div>
            </section>
          )}

          {/* Tab 4: System Analytics & Threshold Sliders Config View */}
          {dashboardTab === 'analytics' && (
            <div className="flex-1 grid grid-cols-12 gap-6 min-h-0 overflow-y-auto">
              
              {/* Left 60%: System Analytics (Real Data Visualizer) */}
              <section className="col-span-7 bg-[#090f1d]/50 border border-cyan-950/50 rounded p-5 flex flex-col gap-6 shadow-xl backdrop-blur-md">
                <div className="border-b border-cyan-950/40 pb-3">
                  <h2 className="text-sm font-bold tracking-wider text-slate-300 uppercase m-0">CLASSIFIER & TRUST ANALYTICS</h2>
                  <span className="text-[10px] text-slate-500">Metrics calculated from recorded, explicitly labeled replay observations</span>
                </div>

                {/* Accuracy over time chart */}
                <div className="h-[140px] w-full bg-slate-950/30 border border-cyan-950/20 p-3 rounded">
                  <span className="text-[9px] text-slate-400 font-mono block mb-1 uppercase">Recall on labeled anomaly events</span>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={accuracyData} margin={{ top: 5, right: 10, left: -25, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity="0.3" />
                      <XAxis dataKey="time" stroke="#475569" fontSize={9} />
                      <YAxis stroke="#475569" fontSize={9} domain={[0, 1]} />
                      <Tooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', fontSize: '9px' }} />
                      <Line type="monotone" dataKey="recall" stroke="#0ea5e9" strokeWidth={1.5} dot={{ r: 3 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Row: Anomaly Type + Aircraft Volume */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="h-[130px] w-full bg-slate-950/30 border border-cyan-950/20 p-3 rounded">
                    <span className="text-[9px] text-slate-400 font-mono block mb-1 uppercase">Anomaly Types Distribution</span>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={typeData} margin={{ top: 5, right: 5, left: -30, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity="0.3" />
                        <XAxis dataKey="type" stroke="#475569" fontSize={7} />
                        <YAxis stroke="#475569" fontSize={8} />
                        <Tooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', fontSize: '8px' }} />
                        <Bar dataKey="count" fill="#f43f5e" radius={[2, 2, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  
                  <div className="h-[130px] w-full bg-slate-950/30 border border-cyan-950/20 p-3 rounded">
                    <span className="text-[9px] text-slate-400 font-mono block mb-1 uppercase">Currently tracked aircraft</span>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={volumeData} margin={{ top: 5, right: 5, left: -25, bottom: 5 }}>
                        <defs>
                          <linearGradient id="colorVol" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#10b981" stopOpacity={0.2}/>
                            <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity="0.3" />
                        <XAxis dataKey="time" stroke="#475569" fontSize={9} />
                        <YAxis stroke="#475569" fontSize={9} />
                        <Tooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', fontSize: '8px' }} />
                        <Area type="monotone" dataKey="volume" stroke="#10b981" fillOpacity={1} fill="url(#colorVol)" strokeWidth={1.5} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Confusion Matrix Detail */}
                <div className="bg-[#0b1220] p-4 rounded border border-cyan-950/30">
                  <span className="text-[10px] text-slate-400 font-bold block mb-3">REPLAY CONFUSION MATRIX FEEDBACK</span>
                  <p className="text-[9px] text-slate-500 mb-3">Only explicitly labeled anomaly observations contribute to TP/FN. FP/TN and precision/F1 require labeled normal observations; unlabeled live traffic is never assumed normal.</p>
                  <div className="grid grid-cols-2 gap-4 text-center font-mono">
                    <div className="bg-[#020617] p-3 rounded border border-cyan-950/20">
                      <div className="text-[9px] text-slate-500">TRUE POSITIVES (TP)</div>
                      <div className="text-xl font-bold text-emerald-400">{replayResult?.true_positives ?? '—'}</div>
                    </div>
                    <div className="bg-[#020617] p-3 rounded border border-cyan-950/20">
                      <div className="text-[9px] text-slate-500">FALSE POSITIVES (FP)</div>
                      <div className="text-xl font-bold text-rose-500">{replayResult?.false_positives ?? '—'}</div>
                    </div>
                    <div className="bg-[#020617] p-3 rounded border border-cyan-950/20">
                      <div className="text-[9px] text-slate-500">TRUE NEGATIVES (TN)</div>
                      <div className="text-xl font-bold text-emerald-400">{replayResult?.true_negatives ?? '—'}</div>
                    </div>
                    <div className="bg-[#020617] p-3 rounded border border-cyan-950/20">
                      <div className="text-[9px] text-slate-500">FALSE NEGATIVES (FN)</div>
                      <div className="text-xl font-bold text-rose-500">{replayResult?.false_negatives ?? '—'}</div>
                    </div>
                  </div>
                </div>
                <div className="bg-[#0b1220] p-4 rounded border border-cyan-950/30">
                  <span className="text-[10px] text-slate-400 font-bold block mb-3">RECORDED EVALUATION RUNS</span>
                  {evaluationRuns.length === 0 ? <p className="text-[10px] text-slate-500">No replay evaluation has been recorded yet.</p> : (
                    <div className="space-y-2 max-h-48 overflow-y-auto">
                      {evaluationRuns.map((run) => (
                        <div key={run.id} className="grid grid-cols-4 gap-2 border-t border-cyan-950/30 pt-2 text-[9px]">
                          <span className="text-slate-400">{new Date(run.run_at).toLocaleString()}</span>
                          <span className="text-cyan-300">{run.model_version}</span>
                          <span>Recall <b className="text-emerald-300">{run.notes.startsWith('Positive-only labeled replay:') && run.recall != null ? `${(run.recall * 100).toFixed(1)}%` : '—'}</b></span>
                          <span className="text-slate-500">{run.notes.startsWith('Positive-only labeled replay:') ? `Labeled: ${run.true_positives + run.false_negatives}` : 'Legacy run · metrics not comparable'}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </section>

              {/* Right 40%: Detection config Sliders */}
              <section className="col-span-5 bg-[#090f1d]/50 border border-cyan-950/50 rounded p-5 flex flex-col gap-6 shadow-xl backdrop-blur-md justify-between">
                <div className="bg-[#0b1220] p-4 rounded border border-amber-900/40">
                  <h3 className="text-[10px] font-bold text-amber-200 m-0 mb-2">LIVE CIRCULAR GEOFENCES</h3>
                  <p className="text-[9px] text-slate-500 mb-3">Operator-defined horizontal reference zones (not official restricted-airspace data; all altitudes). Entry alerts require a prior outside observation and a subsequent live report.</p>
                  <form onSubmit={(event) => void createGeofence(event)} className="grid grid-cols-2 gap-2 text-[10px]">
                    <input required maxLength={80} placeholder="Zone name" value={geofenceForm.name} onChange={(event) => setGeofenceForm({ ...geofenceForm, name: event.target.value })} className="col-span-2 bg-slate-950 border border-slate-700 rounded px-2 py-1.5" />
                    <input required type="number" step="any" min="-90" max="90" placeholder="Latitude" value={geofenceForm.latitude} onChange={(event) => setGeofenceForm({ ...geofenceForm, latitude: event.target.value })} className="bg-slate-950 border border-slate-700 rounded px-2 py-1.5" />
                    <input required type="number" step="any" min="-180" max="180" placeholder="Longitude" value={geofenceForm.longitude} onChange={(event) => setGeofenceForm({ ...geofenceForm, longitude: event.target.value })} className="bg-slate-950 border border-slate-700 rounded px-2 py-1.5" />
                    <input required type="number" step="any" min="0.01" max="500" placeholder="Radius (km)" value={geofenceForm.radius_km} onChange={(event) => setGeofenceForm({ ...geofenceForm, radius_km: event.target.value })} className="bg-slate-950 border border-slate-700 rounded px-2 py-1.5" />
                    <button className="bg-amber-500/90 text-slate-950 font-bold rounded px-2 py-1.5">SAVE ZONE</button>
                  </form>
                  {geofenceMessage && <p role="status" className="text-[9px] text-cyan-300 mt-2">{geofenceMessage}</p>}
                  <div className="mt-3 space-y-1.5 max-h-36 overflow-y-auto">
                    {geofences.length === 0 ? <p className="text-[9px] text-slate-500">No operator-defined zones saved.</p> : geofences.map((zone) => (
                      <div key={zone.id} className="flex justify-between items-center gap-2 text-[9px] bg-slate-950/70 rounded px-2 py-1.5">
                        <span className="truncate text-slate-300">{zone.name} · {zone.radius_km} km · {zone.latitude.toFixed(3)}, {zone.longitude.toFixed(3)}</span>
                        <button type="button" onClick={() => void removeGeofence(zone.id)} aria-label={`Delete ${zone.name}`} className="text-rose-300 hover:text-rose-100">REMOVE</button>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="border-b border-cyan-950/40 pb-3 mb-4">
                    <h2 className="text-sm font-bold tracking-wider text-slate-300 uppercase m-0">AERODYNAMIC CORRELATION THRESHOLDS</h2>
                    <span className="text-[10px] text-slate-500">Customize envelope parameters dynamically</span>
                  </div>

                  {/* Sliders */}
                  <div className="space-y-4 font-mono text-[10px]">
                    {/* Slider 1 */}
                    <div className="space-y-1">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Max Implied Speed:</span>
                        <span className="text-cyan-400 font-bold">{config.max_implied_speed_kmh} km/h</span>
                      </div>
                      <input
                        type="range" min="500" max="2500" step="50"
                        value={config.max_implied_speed_kmh}
                        onChange={(e) => setConfig({ ...config, max_implied_speed_kmh: parseFloat(e.target.value) })}
                        className="w-full accent-cyan-500 bg-slate-950 h-1 rounded cursor-pointer border border-cyan-950/20"
                      />
                    </div>

                    {/* Slider 2 */}
                    <div className="space-y-1">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Duplicate ICAO Distance Limit:</span>
                        <span className="text-cyan-400 font-bold">{config.duplicate_icao_dist_km} km</span>
                      </div>
                      <input
                        type="range" min="5" max="150" step="5"
                        value={config.duplicate_icao_dist_km}
                        onChange={(e) => setConfig({ ...config, duplicate_icao_dist_km: parseFloat(e.target.value) })}
                        className="w-full accent-cyan-500 bg-slate-950 h-1 rounded cursor-pointer border border-cyan-950/20"
                      />
                    </div>

                    {/* Slider 3 */}
                    <div className="space-y-1">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Max climb envelope rate:</span>
                        <span className="text-cyan-400 font-bold">{config.max_vertical_rate_ms} m/s</span>
                      </div>
                      <input
                        type="range" min="10" max="150" step="5"
                        value={config.max_vertical_rate_ms}
                        onChange={(e) => setConfig({ ...config, max_vertical_rate_ms: parseFloat(e.target.value) })}
                        className="w-full accent-cyan-500 bg-slate-950 h-1 rounded cursor-pointer border border-cyan-950/20"
                      />
                    </div>

                    {/* Slider 4 */}
                    <div className="space-y-1">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Max Ground roll altitude:</span>
                        <span className="text-cyan-400 font-bold">{config.max_ground_altitude_m} m</span>
                      </div>
                      <input
                        type="range" min="10" max="500" step="10"
                        value={config.max_ground_altitude_m}
                        onChange={(e) => setConfig({ ...config, max_ground_altitude_m: parseFloat(e.target.value) })}
                        className="w-full accent-cyan-500 bg-slate-950 h-1 rounded cursor-pointer border border-cyan-950/20"
                      />
                    </div>

                    {/* Slider 5 */}
                    <div className="space-y-1">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Max Ground roll speed limit:</span>
                        <span className="text-cyan-400 font-bold">{config.max_ground_speed_ms} m/s</span>
                      </div>
                      <input
                        type="range" min="10" max="150" step="5"
                        value={config.max_ground_speed_ms}
                        onChange={(e) => setConfig({ ...config, max_ground_speed_ms: parseFloat(e.target.value) })}
                        className="w-full accent-cyan-500 bg-slate-950 h-1 rounded cursor-pointer border border-cyan-950/20"
                      />
                    </div>

                    {/* Slider 6 */}
                    <div className="space-y-1">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Min flight speed required:</span>
                        <span className="text-cyan-400 font-bold">{config.min_flight_speed_ms} m/s</span>
                      </div>
                      <input
                        type="range" min="5" max="100" step="5"
                        value={config.min_flight_speed_ms}
                        onChange={(e) => setConfig({ ...config, min_flight_speed_ms: parseFloat(e.target.value) })}
                        className="w-full accent-cyan-500 bg-slate-950 h-1 rounded cursor-pointer border border-cyan-950/20"
                      />
                    </div>
                  </div>
                </div>

                {/* Operations buttons */}
                <div className="space-y-3 pt-6 border-t border-cyan-950/40">
                  {/* Replay indicator summary */}
                  {replayResult && (
                    <div className="bg-[#020617] border border-cyan-950 p-2.5 rounded text-[9px] leading-relaxed text-cyan-500 font-mono space-y-1">
                      <div className="font-bold border-b border-cyan-950/40 pb-1 text-slate-300">LIVE REPLAY EVALUATION RESULTS:</div>
                      <div>MODEL VERIFICATION SCORE: <b>{replayResult.model_version}</b></div>
                      <div>PRECISION: <b className="text-slate-400">{replayResult.precision == null ? 'NOT MEASURABLE — no labeled normal observations' : `${(replayResult.precision * 100).toFixed(2)}%`}</b></div>
                      <div>RECALL: <b className="text-emerald-400">{replayResult.recall == null ? 'NOT MEASURABLE — no labeled anomalies' : `${(replayResult.recall * 100).toFixed(2)}%`}</b></div>
                      <div>F1: <b className="text-slate-400">{replayResult.f1 == null ? 'NOT MEASURABLE — requires both label classes' : `${(replayResult.f1 * 100).toFixed(2)}%`}</b></div>
                    </div>
                  )}

                  <button
                    onClick={handleReplaySession}
                    disabled={isReplaying}
                    className="w-full bg-cyan-600 hover:bg-cyan-500 disabled:bg-cyan-900 text-black font-bold text-xs py-2.5 px-4 rounded transition-all font-mono shadow-md hover:shadow-cyan-500/25"
                  >
                    {isReplaying ? "EVALUATING STORED TELEMETRY..." : "EVALUATE LABELED STORED TELEMETRY"}
                  </button>

                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <button
                      onClick={handleSaveConfig}
                      className="border border-cyan-950 bg-slate-950 hover:bg-cyan-950/10 text-slate-300 font-bold py-2 px-4 rounded transition-all font-mono"
                    >
                      SAVE CONFIG
                    </button>
                    <button
                      onClick={handleResetConfig}
                      className="border border-rose-950 bg-slate-950 hover:bg-rose-950/10 text-rose-400 font-bold py-2 px-4 rounded transition-all font-mono"
                    >
                      RESET TO DEFAULTS
                    </button>
                  </div>
                </div>
              </section>

            </div>
          )}

          {/* --- Bottom 3 Stat Cards --- */}
          <section className="grid grid-cols-3 gap-6 mt-6">
            
            {/* Card 1 */}
            <div className="p-4 border border-cyan-950/40 bg-[#060b14]/50 rounded shadow-lg backdrop-blur-sm flex flex-col justify-between">
              <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Total Tracked Targets</span>
              <div className="flex items-baseline gap-2 mt-2">
                <span className="text-3xl font-black text-slate-100">{stats.total}</span>
                <span className="text-[10px] text-emerald-400 font-semibold">Active Feeds</span>
              </div>
            </div>

            {/* Card 2 */}
            <div className="p-4 border border-cyan-950/40 bg-[#060b14]/50 rounded shadow-lg backdrop-blur-sm flex flex-col justify-between">
              <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Security Anomalies</span>
              <div className="flex items-baseline gap-2 mt-2">
                <span className="text-3xl font-black text-rose-500">{stats.anomalies}</span>
                <span className="text-[10px] text-rose-400 font-semibold">Flagged / Suppressed</span>
              </div>
            </div>

            {/* Card 3 */}
            <div className="p-4 border border-cyan-950/40 bg-[#060b14]/50 rounded shadow-lg backdrop-blur-sm flex flex-col justify-between">
              <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Station Trust Rating</span>
              <div className="flex items-baseline gap-2 mt-2">
                <span className="text-3xl font-black text-cyan-400">{stats.avgTrust == null ? '—' : `${stats.avgTrust}%`}</span>
                <span className="text-[10px] text-cyan-400 font-semibold">Reliability Index</span>
              </div>
            </div>

          </section>

        </div>
      )}

      {/* --- Footer --- */}
      <footer className="border-t border-cyan-950/40 bg-[#060b14] px-6 py-3 flex items-center justify-between text-[10px] text-slate-500 relative z-10">
        <div>AirGuard Ground Station Receiver v0.1.0</div>
        <div className="flex items-center gap-4">
          <a href="docs/MODEL_CARD.md" className="hover:text-slate-300 transition-colors">Model Card</a>
          <a href="docs/ARCHITECTURE_DECISIONS.md" className="hover:text-slate-300 transition-colors">ADR Logs</a>
          <a href="docs/DEMO_SCRIPT.md" className="hover:text-slate-300 transition-colors">Demo Script</a>
        </div>
      </footer>
    </div>
  );
}
