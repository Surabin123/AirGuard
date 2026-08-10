# Research Paper: AirGuard - A Real-Time ADS-B Trust-Scoring Ground Station

## 1. Abstract
The Automatic Dependent Surveillance-Broadcast (ADS-B) protocol is a critical component of modern aviation monitoring, enabling aircraft to broadcast their identity, position, and velocity. However, the protocol lacks built-in encryption and authentication mechanisms, making it highly susceptible to cyberattacks such as spoofing, message injection, and tampering. This paper proposes a real-time ADS-B trust-scoring system designed to detect and mitigate these vulnerabilities. The primary objective is to evaluate incoming flight data streams and assign a reliability metric, or "trust score," using a hybrid machine learning approach. By integrating a supervised soft-voting ensemble (Random Forest and Gradient Boosting) with an unsupervised Autoencoder, the system can identify both known attack signatures and novel anomalies. The expected outcome is a robust, scalable framework that significantly enhances the situational awareness and security of air traffic control systems.

## 2. Introduction
The aviation industry relies heavily on situational awareness to maintain safe distances between aircraft and manage airspace efficiently. ADS-B has largely replaced traditional radar due to its high precision and frequent update rates. Despite its operational advantages, the unencrypted nature of ADS-B transmissions allows malicious actors to easily intercept and broadcast false flight data using inexpensive software-defined radios (SDRs). 

The problem addressed in this research is the lack of native data validation in the ADS-B protocol. Current air traffic management infrastructure implicitly trusts the broadcasted data, which can lead to catastrophic consequences if the data is manipulated. The primary objective of this project is to develop a secondary verification layer that operates independently of the aircraft's avionics. The motivation stems from the increasing accessibility of radio spoofing hardware and the critical need to secure civilian aviation networks against digital interference.

## 3. Literature Survey
Recent studies have highlighted the structural vulnerabilities of the ADS-B protocol.
1. **Cryptographic Approaches:** Several researchers have proposed adding cryptographic signatures to ADS-B messages. While effective in theory, these approaches require massive overhauls of existing global aviation hardware, making immediate adoption impractical.
2. **Multilateration (MLAT):** Existing systems often use Time Difference of Arrival (TDOA) to verify an aircraft's physical location. While reliable, MLAT requires a dense network of synchronized ground receivers, which is expensive and unfeasible over oceans or remote areas.
3. **Machine Learning Anomaly Detection:** Recent works have applied standard Support Vector Machines (SVM) and basic neural networks to detect flight path deviations. These methods often struggle with high false-positive rates during legitimate, sudden maneuvers (e.g., collision avoidance) and fail to generalize to previously unseen attack vectors.

## 4. Research Gap
Existing solutions either require impossible hardware retrofits or rely on single-layered detection algorithms that produce excessive false alarms. Furthermore, most machine learning approaches in current literature focus strictly on supervised learning, which fundamentally limits their ability to detect zero-day spoofing techniques. There is a distinct gap in literature regarding hybrid models that can simultaneously recognize known tampering patterns and detect statistical outliers in real-time, low-latency environments.

## 5. Proposed Methodology
To address these limitations, we propose a hybrid detection architecture. The system workflow begins with continuous ingestion of live ADS-B data via the OpenSky Network. 
The proposed solution utilizes a dual-engine pipeline:
1. **Supervised Engine:** A soft-voting ensemble comprising Random Forest and Gradient Boosting models. This component is trained on historical data containing labeled examples of known attacks (e.g., position jumps, ghost aircraft injection).
2. **Unsupervised Engine:** A PyTorch-based Autoencoder that reconstructs nominal flight behavior. If an incoming signal deviates significantly from the reconstructed baseline, it is flagged as anomalous.
The final trust score is a weighted aggregation of the outputs from both engines, allowing the system to handle both historical threats and novel deviations effectively.

## 6. System Architecture
The overall architecture consists of three primary modules:
1. **Data Ingestion Module:** A continuous polling mechanism that interfaces with external APIs to fetch raw trajectory data, storing it in an asynchronous PostgreSQL database.
2. **Processing & Detection Module:** The core computational engine where the hybrid ML models evaluate the incoming telemetry.
3. **Visualization Module:** A real-time dashboard that plots aircraft coordinates on a 3D globe and highlights anomalous behavior using color-coded trust metrics.

*Data Flow:* Raw data -> Ingestion Pipeline -> Database -> Hybrid ML Engine -> Trust Score Generation -> REST API -> Client Dashboard.

## 7. Algorithms/Techniques
The methodology relies on the following specific techniques:
- **Random Forest & Gradient Boosting (Supervised):** These algorithms evaluate discrete features such as velocity changes, altitude drop rates, and transmission frequencies. Soft-voting is utilized to average the probability outputs of both classifiers, yielding a smoother confidence metric rather than a binary classification.
- **Autoencoder (Unsupervised):** An artificial neural network designed to learn efficient data encodings in an unsupervised manner. The model compresses the flight telemetry into a lower-dimensional latent space and reconstructs it. The reconstruction error (Mean Squared Error) serves as the anomaly score; high error indicates behavior that the model has not seen during nominal training phases.

## 8. Expected Outcomes
The proposed system is expected to process high-throughput flight data with minimal latency, making it suitable for real-time monitoring. The hybrid model approach aims to reduce false-positive rates by at least 15% compared to isolated supervised models. Practically, this framework will provide air traffic controllers and researchers with a visual, intuitive assessment of signal integrity, acting as an immediate software-based defense against ADS-B spoofing without requiring hardware modifications to existing aircraft.

## 9. References
[1] Strohmeier, M., Schäfer, M., Lenders, V., & Martinovic, I. (2014). "Realities and challenges of nextgen air traffic management: the case of ADS-B." IEEE Communications Magazine, 52(5), 111-118.
[2] Schäfer, M., Lenders, V., & Martinovic, I. (2013). "Experimental analysis of attacks on next generation air traffic communication." In International Conference on Applied Cryptography and Network Security (pp. 253-271). Springer, Berlin, Heidelberg.
[3] Ying, X., Mazer, J., Bernieri, G., et al. (2019). "Detecting ADS-B spoofing attacks using deep neural networks." IEEE Conference on Communications and Network Security (CNS).
