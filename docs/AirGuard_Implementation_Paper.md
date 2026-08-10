# Implementation Paper: AirGuard - A Real-Time ADS-B Trust-Scoring Ground Station

## 1. Abstract
The security of aviation communication protocols is a growing concern, particularly regarding the unencrypted Automatic Dependent Surveillance-Broadcast (ADS-B) system. This paper details the implementation of AirGuard, a real-time trust-scoring ground station designed to detect malicious manipulation of flight data. The implemented solution leverages a Python-based FastAPI backend, a PostgreSQL database, and a React-based 3D visualization dashboard. The core contribution is a dual-engine anomaly detection pipeline utilizing Scikit-learn and PyTorch to assign trust scores to live flight telemetry. The resulting system successfully processes continuous streams of data, identifying simulated injection and tampering attacks with low latency, thereby demonstrating a viable, software-only defense mechanism for modern air traffic monitoring.

## 2. Introduction
With the global mandate for ADS-B outfitting, civil aviation heavily relies on open radio broadcasts for air traffic control. The lack of cryptographic authentication in ADS-B messages exposes the network to spoofing and tampering. The objective of this project was to construct a functional software prototype capable of auditing these signals in real-time. The implemented solution, AirGuard, acts as an independent ground station that evaluates the physical and statistical consistency of flight trajectories, providing immediate visual feedback regarding the reliability of the tracked aircraft.

## 3. System Design & Architecture
The system architecture follows a modern, decoupled client-server model.
- **Hardware/Software Requirements:** The system is containerized using Docker, requiring only a host capable of running Docker Desktop. Development was conducted using Python 3.11 and Node.js 18+.
- **System Workflow:**
  1. The backend continuously polls the OpenSky Network API for live flight vectors.
  2. Data is sanitized and pushed into a PostgreSQL database using asynchronous SQLAlchemy drivers.
  3. A background detection loop pulls recent trajectories and feeds them into the machine learning models.
  4. The models calculate an aggregated trust score, updating the database.
  5. The frontend polls the backend via REST APIs to render the aircraft on a 3D interface, coloring them based on their trust scores.

## 4. Implementation Details
The implementation is modularized into distinct components:
- **Backend Implementation:** Developed using FastAPI to handle asynchronous requests and background tasks efficiently. Rate limiting is enforced via `slowapi` to prevent API abuse. The data layer utilizes `asyncpg` for non-blocking database transactions.
- **AI/ML Implementation:** The detection service instantiates two distinct models. The supervised ensemble is implemented using `scikit-learn`, combining Random Forest and Gradient Boosting classifiers. The unsupervised component is a custom Autoencoder built with `PyTorch`. 
- **Frontend Implementation:** The user interface is built with React 18 and Vite. State management across the application is handled by `zustand`. 
- **Visualization:** A 3D geospatial interface is rendered using `cesium` and its React wrapper `resium`, allowing for accurate global tracking. `recharts` is integrated to display real-time metric graphs and anomaly panels.

## 5. Algorithms/Code Logic
The critical processing logic resides in the trust score calculation algorithm:
1. **Feature Extraction:** Raw velocity, heading, and altitude data are transformed into derivative features (e.g., acceleration, rate of climb).
2. **Supervised Evaluation:** The ensemble model outputs a probability score $P_s$ indicating the likelihood of the trajectory belonging to a known attack class.
3. **Unsupervised Evaluation:** The Autoencoder reconstructs the input vector. The reconstruction error is normalized to a probability score $P_u$.
4. **Aggregation:** The final trust score $T$ is calculated as $T = 100 - (w_1 \cdot P_s + w_2 \cdot P_u)$, where $w_1$ and $w_2$ are tuned weights.
This logic runs in an asynchronous background loop, ensuring the API remains responsive.

## 6. Tools and Technologies Used
- **Programming Languages:** Python 3.11, TypeScript.
- **Backend Frameworks:** FastAPI, SQLAlchemy, Alembic.
- **Frontend Frameworks:** React 18, Vite, Tailwind CSS.
- **Database:** PostgreSQL (with asyncpg).
- **AI/ML Tools:** Scikit-learn, PyTorch, SHAP (for model interpretability).
- **Mapping & Data Visualization:** Cesium, Resium, Recharts.
- **Environment & Deployment:** Docker, Docker Compose, Poetry (dependency management).

## 7. Results and Discussion
The implemented system was tested under simulated attack conditions using custom injection scripts (`inject_anomaly.py`). 
- **Performance:** The FastAPI backend consistently processed incoming batches of flight data with a polling latency averaging under 150ms. 
- **Detection Accuracy:** During local testing, the hybrid model successfully flagged simulated position jumps and anomalous altitude variations. The soft-voting ensemble effectively reduced false positives during high-acceleration maneuvers compared to using a single decision tree.
- **Visualization:** The Resium frontend smoothly rendered the data, immediately reflecting trust score degradations by changing the aircraft marker colors to indicate warning states.

## 8. Conclusion
The AirGuard project successfully demonstrates the feasibility of a software-based verification layer for ADS-B data. By combining a robust asynchronous backend with a hybrid machine learning pipeline, the system meets its objective of identifying anomalous flight behavior in real-time. The major contribution is the seamless integration of unsupervised and supervised learning models into a cohesive, easily deployable ground station architecture.

## 9. Future Scope
Future iterations of AirGuard can be expanded in several directions:
- **Direct SDR Integration:** Bypassing web APIs to ingest raw baseband radio signals directly from a Software Defined Radio.
- **Scalability:** Transitioning the PostgreSQL database to a distributed time-series database (e.g., TimescaleDB) to handle higher volumes of global traffic.
- **Advanced Models:** Incorporating Recurrent Neural Networks (LSTMs) to better capture the temporal dependencies in long-term flight paths.

## 10. References
[1] FastAPI Documentation, "FastAPI framework, high performance, easy to learn, fast to code, ready for production," 2023. [Online]. Available: https://fastapi.tiangolo.com/
[2] Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research, 12, 2825-2830.
[3] Paszke, A. et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." In Advances in Neural Information Processing Systems 32. Curran Associates, Inc.
[4] OpenSky Network, "The OpenSky Network API," 2023. [Online]. Available: https://opensky-network.org/apidoc/
