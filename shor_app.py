import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from fractions import Fraction
import pandas as pd
import time
from math import ceil, log2, gcd
from qiskit.quantum_info import Operator

# Set page config
st.set_page_config(page_title="Shor's Algorithm Visualizer", layout="wide")

# Custom CSS for Premium Look & Animation
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&display=swap');
    
    html, body, [class*="st-"] { font-family: "Outfit", "Segoe UI", sans-serif; }
    h1, h2, h3 { font-weight: 700; color: #000000; }

    /* Hide standard error icon/box if explicitly requested via class */
    [data-testid="stExpanderIconError"] {
        display: none !important;
    }

    /* Hide Streamlit Deploy button */
    .stAppDeployButton {
        display: none !important;
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    /* Centered Blackout/Instruction Overlay */
    .initial-overlay {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 60vh;
        text-align: center;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 3rem;
        border: 2px dashed #3B82F6;
        animation: fadeIn 0.8s ease-out;
    }

    .blackout-overlay {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 60vh;
        text-align: center;
        background: #111827;
        color: #9CA3AF;
        border-radius: 20px;
        padding: 3rem;
        border: 2px dashed #EF4444;
        animation: fadeIn 0.8s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Pulsing Loading Animation */
    .loading-pulse {
        width: 100px;
        height: 100px;
        background: #3B82F6;
        border-radius: 50%;
        display: inline-block;
        position: relative;
        box-sizing: border-box;
        animation: pulse 1.5s linear infinite;
    }
    .loading-pulse::after {
        content: '';  
        box-sizing: border-box;
        width: 100px;
        height: 100px;
        border-radius: 50%;
        border: 5px solid #3B82F6;
        position: absolute;
        left: 0;
        top: 0;
        animation: animloader 1.5s linear infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0); opacity: 1; }
        100% { transform: scale(1); opacity: 0; }
    }
    @keyframes animloader {
        0% { transform: scale(0); opacity: 1; }
        100% { transform: scale(1.5); opacity: 0; }
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Shor's Algorithm: Quantum Scaling & Factoring")

# Sidebar
st.sidebar.header("⚙️ Experiment Parameters")
# Start with None (Blank)
N_input = st.sidebar.number_input("Number to factor (N)", min_value=3, max_value=10000, value=None, step=1, placeholder="Enter N (e.g. 15)")

def get_coprimes(n):
    if not n: return []
    if n > 200: return [2, 3, 7, 11, 13, 17, 19, 23]
    return [i for i in range(2, n) if gcd(i, n) == 1]

coprimes = get_coprimes(N_input)
a_input = st.sidebar.selectbox("Coprime (a)", coprimes, index=None, placeholder="Choose a coprime...", help="Only valid coprimes are shown.")

# --- Helper Functions ---
def find_period_classical(a, N, timeout=45):
    if not a: return None, 0
    start = time.perf_counter()
    for r in range(1, N + 1):
        if time.perf_counter() - start > timeout:
            return "TIMEOUT", time.perf_counter() - start
        if pow(int(a), int(r), int(N)) == 1:
            return r, time.perf_counter() - start
    return None, 0

def qpe_optimized(a, N, n_count=3):
    n_qubits = ceil(log2(N))
    matrix = np.zeros((2**n_qubits, 2**n_qubits))
    for i in range(2**n_qubits):
        if i < N: matrix[(i * a) % N, i] = 1
        else: matrix[i, i] = 1
    U_gate = Operator(matrix).to_instruction()
    qc = QuantumCircuit(n_count + n_qubits, n_count)
    for q in range(n_count): qc.h(q)
    qc.x(n_count)
    for q in range(n_count):
        qc.append(U_gate.repeat(2**q).control(), [q] + list(range(n_count, n_count + n_qubits)))
    from qiskit.circuit.library import QFT
    qc.append(QFT(n_count).inverse(), range(n_count))
    qc.measure(range(n_count), range(n_count))
    return qc

def quantum_emulator(r, N, n_shots=1024):
    n_count = int(2 * ceil(log2(N)))
    counts = {}
    if not r: return counts, n_count
    ideal_phases = [(k / r) % 1 for k in range(r)]
    for _ in range(n_shots):
        phase = np.random.choice(ideal_phases)
        decimal = int(round(phase * (2**n_count))) % (2**n_count)
        binary = format(decimal, f'0{n_count}b')
        counts[binary] = counts.get(binary, 0) + 1
    return counts, n_count

# --- EXECUTION ---
r_truth, c_time, q_counts, q_duration, q_n_count = None, 0, None, 0, 3
n_count, n_shots = 3, 1024 # Defaults
sim_timeout_triggered = False

# Input Validation
if N_input is not None and N_input < 3:
    st.error("🚀 **Input Error:** N must be at least 3 to perform factoring.")
    st.stop()

if N_input and a_input:
    if gcd(int(a_input), int(N_input)) != 1:
        st.error(f"❌ **Coprimality Error:** The base 'a' ({a_input}) and 'N' ({N_input}) are not coprime (gcd != 1). Shor's algorithm strictly requires coprime inputs.")
        st.stop()
    
    # Scientific parameter calculation
    n_count = 2 * ceil(log2(N_input)) if N_input > 4 else 3
    q_n_count = n_count
    n_shots = max(2048, 4 * N_input)
    
    # Modern Loading State
    with st.status("🚀 Initializing Quantum Simulation...", expanded=True) as status:
        st.write("🔍 Calculating classical period...")
        start_time = time.perf_counter()
        r_truth, c_time = find_period_classical(a_input, N_input, timeout=45)
        
        if r_truth == "TIMEOUT":
            sim_timeout_triggered = True
            status.update(label="❌ Simulation Failed (Timeout)", state="error", expanded=False)
        elif r_truth:
            st.write("⚛️ Constructing quantum interference pattern...")
            if N_input <= 32:
                sim_shots = 512  # Optimized for performance while maintaining clarity
                st.write(f"⚛️ Building QPE circuit ({n_count} counting qubits, {sim_shots} shots)...")
                start_q = time.perf_counter()
                
                # Check for timeout before starting heavy simulation
                if time.perf_counter() - start_time > 40:
                    sim_timeout_triggered = True
                else:
                    qc = qpe_optimized(a_input, N_input, n_count=n_count)
                    backend = Aer.get_backend('aer_simulator')
                    
                    # Transpilation can be slow for large circuits
                    t_qc = transpile(qc, backend)
                    
                    if time.perf_counter() - start_time > 43:
                        sim_timeout_triggered = True
                    else:
                        result = backend.run(t_qc, shots=sim_shots).result()
                        q_counts = result.get_counts()
                        q_duration = time.perf_counter() - start_q
                        n_shots = sim_shots  # Update n_shots if used downstream
            else:
                emu_shots = 2048  # Higher shots for cheap emulator
                st.write(f"🧮 Emulating ideal QPE sampling ({emu_shots} shots)...")
                start_q = time.perf_counter()
                q_counts, q_n_count = quantum_emulator(r_truth, N_input, n_shots=emu_shots)
                q_duration = time.perf_counter() - start_q
                n_shots = emu_shots
            
            if r_truth is None:
                st.warning("⚠️ **Search Warning:** A valid period 'r' was not found within the computational limits. This happens for certain (a, N) combinations or if the period is extremely large.")

            # Final check for total timeout
            if time.perf_counter() - start_time > 45 or sim_timeout_triggered:
                sim_timeout_triggered = True
                status.update(label="❌ Simulation Failed (Timeout)", state="error", expanded=False)
            else:
                status.update(label="✅ Simulation Complete!", state="complete", expanded=False)
        else:
            status.update(label="✅ Simulation Complete!", state="complete", expanded=False)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📈 Performance Scaling", "🏫 Education & Theory"])

with tab1:
    if not N_input or not a_input:
        st.markdown("""
            <div class="initial-overlay">
                <span class="loading-pulse"></span>
                <h1 style="margin-top:2rem; color:#1E3A8A;">Awaiting Quantum Parameters</h1>
                <p style="font-size:1.2rem; color:#4B5563; max-width:600px;">
                    Please use the <b>sidebar on the left</b> to enter a value for <b>N</b> (the number to factor) 
                    and select a coprime <b>a</b>. Once selected, the quantum simulation will automatically begin.
                </p>
                <div style="margin-top:1rem; padding:1rem; background:#EFF6FF; border-radius:10px; color:#1E40AF;">
                    👈 <b>Action required:</b> Click the arrow at the top-left if the sidebar is hidden.
                </div>
            </div>
        """, unsafe_allow_html=True)
    elif sim_timeout_triggered:
        st.markdown("""
            <div class="blackout-overlay" style="border-color: #FACC15; background: #1E1B4B;">
                <h2 style="color: #FACC15; margin-bottom: 1rem;">⏳ Simulation Timeout</h2>
                <p style="font-size: 1.1rem; max-width: 700px; color: #E5E7EB;">
                    The simulation could not generate results because it exceeded the 45-second limit. 
                    This is likely due to an error or heavy complexity in the <b>N value</b> and <b>cofactor</b> provided.
                </p>
                <div style="margin-top: 1.5rem; padding: 1rem; border: 1px solid #FACC15; border-radius: 8px; color: #FDE047;">
                    💡 <b>Recommendation:</b> Please try another N and coprime combination.
                </div>
            </div>
        """, unsafe_allow_html=True)
    elif r_truth is None:
        st.markdown("""
            <div class="blackout-overlay">
                <h2 style="color: #EF4444; margin-bottom: 1rem;">🚨 Period Not Found</h2>
                <p style="font-size: 1.1rem; max-width: 700px; color: #9CA3AF;">
                    The classical search could not find a stable period for $N={N_input}$ and $a={a_input}$ within the simulation range. 
                    This usually happens if the combination of parameters creates a pattern that is too long or non-periodic for the current algorithm to detect.
                </p>
                <div style="margin-top: 1.5rem; padding: 1rem; border: 1px solid #374151; border-radius: 8px; color: #F87171;">
                    💡 <b>Advice:</b> Try a different coprime value from the sidebar.
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        q_steps = ceil(log2(N_input))**2
        reduction = r_truth / q_steps if q_steps > 0 else 1
        st.markdown(f"""<div style="background:#F3F4F6;padding:1rem;border-radius:12px;border-left:8px solid #1E3A8A;margin-bottom:2rem;color:black;"><h3 style="margin:0;color:black;">🏁 Correct Answer: Period (r) = {r_truth}</h3><p style="margin:0.5rem 0 0 0;color:black;">🚀 <b>Speedup Factor:</b> {reduction:.2f}x fewer steps than classical.</p></div>""", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🕰️ Classical Side")
            x, y = np.arange(26), [pow(int(a_input), int(i), int(N_input)) for i in range(26)]
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(x, y, 'o-', color='#3B82F6')
            ax.axvline(x=r_truth, color='r', linestyle='--')
            ax.set_title(f"$f(x) = {a_input}^x$ mod ${N_input}$")
            st.pyplot(fig)
            st.write(f"Time: **{c_time*1000:.4f} ms** | Steps: **{r_truth}**")
            st.markdown("> **Energy Note**: For $N > 2^{{1024}}$, classical computing would require the energy of a city to factor this, while a QPU uses roughly the energy of a lightbulb.")
            
        with col2:
            st.subheader(f"⚛️ Quantum Side {'[⚡ Sim]' if N_input <= 32 else '[🧮 Emulator]'}")
            if q_counts:
                fig2, ax2 = plt.subplots(figsize=(8,4))
                plot_histogram(q_counts, ax=ax2)
                st.pyplot(fig2)
                match_counts = sum(v for k,v in q_counts.items() if Fraction(int(k,2)/(2**q_n_count)).limit_denominator(N_input).denominator == r_truth or (r_truth % Fraction(int(k,2)/(2**q_n_count)).limit_denominator(N_input).denominator == 0 and Fraction(int(k,2)/(2**q_n_count)).limit_denominator(N_input).denominator > 1))
                st.write(f"Time: **{q_duration*1000:.4f} ms** | Effort: **{q_steps}** logical ops.")
                st.success(f"**Success Prob:** {(match_counts/n_shots)*100:.2f}%")
            else:
                st.warning("Quantum results unavailable.")

with tab2:
    st.header("⏱️ Performance Analysis")
    b = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    c_t = 10**(-10) * np.exp(0.06 * b)
    q_t = 10**(-3) * (b**2)
    s_b = np.array([4, 8, 16, 32, 64])
    s_t = 10**(-5) * (2.0**s_b)
    fig3, ax3 = plt.subplots(figsize=(10,5))
    ax3.plot(b, c_t, 'r-x', label="Classical (GNFS)")
    ax3.plot(b, q_t, 'g-o', label="Quantum QPU")
    ax3.plot(s_b, s_t, 'y--', label="Laptop Sim")
    ax3.set_yscale('log'); ax3.set_xlabel("Bits"); ax3.set_ylabel("Seconds"); ax3.legend(); ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

with tab3:
    st.header("📖 Quantum Theory & Comparative Analysis")
    st.write("""
    Shor's algorithm is the "Holy Grail" of quantum computing. It demonstrates a mathematical shortcut that is physically impossible 
    on classical hardware. While classical computers must check every factor one-by-one, quantum computers use **Interference** 
    to cancel out wrong answers and find the correct period $r$ in a single shot.
    """)

    st.subheader("🏁 10-Point Comparison: Classical vs. Quantum")
    
    comp_10_data = {
        "Feature": [
            "Core Strategy",
            "Mathematical Basis",
            "Scaling Complexity",
            "RSA-2048 Security",
            "Best Algorithm",
            "Energy Efficiency",
            "Error Tolerance",
            "Infrastructure",
            "Calculation Type",
            "Real-World Status"
        ],
        "Classical Computer": [
            "Brute-force or GNFS (Sieve)",
            "Number Theory (Factoring)",
            "Exponential: O(exp(n^1/3))",
            "Secure (Millions of Years)",
            "General Number Field Sieve",
            "Poor (Energy of a city)",
            "High (Deterministic bits)",
            "Transistors & Logic Gates",
            "Sequential (Step-by-step)",
            "Production Ready (Mature)"
        ],
        "Quantum Computer": [
            "Quantum Phase Estimation (QPE)",
            "Linear Algebra (Hilbert Space)",
            "Polynomial: O(n^2)",
            "Vulnerable (Minutes)",
            "Shor's Algorithm",
            "Excellent (Energy of a lightbulb)",
            "Low (Requires Error Correction)",
            "Qubits & Entanglement",
            "Parallel (Massive Superposition)",
            "Experimental (Nascent)"
        ]
    }
    
    st.table(pd.DataFrame(comp_10_data))

    st.markdown(r"""
    ### 🧠 The Modular Link
    Shor's algorithm works by transforming a **Factoring Problem** into a **Period-Finding Problem**. 
    Finding the factor $N$ is hard, but finding the period $r$ of $a^x \pmod N$ is something a quantum computer is naturally 
    designed to do using the Quantum Fourier Transform (QFT).
    """)
