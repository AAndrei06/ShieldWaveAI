import joblib
import numpy as np
from scapy.all import sniff, IP, TCP, UDP

# Încarcă modelul și scalerul salvate anterior
model = joblib.load('random_forest_nsl_kdd.pkl')
scaler = joblib.load('scaler.pkl')

def extract_features(packet):
    features = {}

    if packet.haslayer(IP):
        # Extragerea caracteristicilor de bază
        features['protocol_type'] = packet[IP].proto  # 1
        features['src_ip'] = packet[IP].src
        features['dst_ip'] = packet[IP].dst
        features['src_bytes'] = len(packet)  # 4
        features['dst_bytes'] = len(packet)  # 5

        if packet.haslayer(TCP):
            # Asigură-te că câmpul 'urg' există în TCP
            try:
                features['urgent'] = packet[TCP].urg  # 6
            except AttributeError:
                features['urgent'] = 0  # Setează la 0 dacă câmpul nu există
        else:
            features['urgent'] = 0  # Dacă nu este TCP, setăm 0

        if packet.haslayer(TCP) or packet.haslayer(UDP):
            features['num_failed_logins'] = 0  # Exemplu pentru un calcul personalizat de caracteristici (7)
            features['count'] = 1  # 8

            # Alte caracteristici
            features['wrong_fragment'] = 0  # Exemplu
            features['hot'] = 0  # Exemplu
            features['logged_in'] = 0  # Exemplu
            features['num_compromised'] = 0  # Exemplu
            features['root_shell'] = 0  # Exemplu
            features['su_attempted'] = 0  # Exemplu
            features['num_root'] = 0  # Exemplu
            features['num_file_creations'] = 0  # Exemplu
            features['num_shells'] = 0  # Exemplu
            features['num_access_files'] = 0  # Exemplu
            features['num_outbound_cmds'] = 0  # Exemplu
            features['is_host_login'] = 0  # Exemplu
            features['is_guest_login'] = 0  # Exemplu
            features['count'] = 1  # Exemplu
            features['srv_count'] = 1  # Exemplu
            features['serror_rate'] = 0  # Exemplu
            features['srv_serror_rate'] = 0  # Exemplu
            features['rerror_rate'] = 0  # Exemplu
            features['srv_rerror_rate'] = 0  # Exemplu
            features['same_srv_rate'] = 0  # Exemplu
            features['diff_srv_rate'] = 0  # Exemplu
            features['srv_diff_host_rate'] = 0  # Exemplu
            features['dst_host_count'] = 1  # Exemplu
            features['dst_host_srv_count'] = 1  # Exemplu
            features['dst_host_same_srv_rate'] = 0  # Exemplu
            features['dst_host_diff_srv_rate'] = 0  # Exemplu
            features['dst_host_same_src_port_rate'] = 0  # Exemplu
            features['dst_host_srv_diff_host_rate'] = 0  # Exemplu
            features['dst_host_serror_rate'] = 0  # Exemplu
            features['dst_host_srv_serror_rate'] = 0  # Exemplu
            features['dst_host_rerror_rate'] = 0  # Exemplu
            features['dst_host_srv_rerror_rate'] = 0  # Exemplu

    return features



# Procesarea unui pachet individual
def process_packet(packet):
    features = extract_features(packet)
    if features is None:
        return

    # Ordinea exactă a caracteristicilor (41, ca în NSL-KDD)
    feature_order = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]

    vector = [features.get(k, 0) for k in feature_order]
    vector = np.array(vector).reshape(1, -1)

    # Normalizare
    vector = scaler.transform(vector)

    # Predicție
    prediction = model.predict(vector)[0]
    label = "Normal" if prediction == 0 else "Anomaly"
    print(f"[{label}] Packet detected.")

# Captură pachete în timp real
def start_sniffing(interface="wlan0"):
    print(f"Listening on {interface}...")
    sniff(iface=interface, prn=process_packet, store=0)

# Punct de intrare
if __name__ == "__main__":
    start_sniffing("wlan0")
