import time
import torch
import pennylane as qml

N = 8
batch_size = 128
inputs = torch.randn(batch_size, 2**N)
inputs = torch.nn.functional.normalize(inputs, p=2, dim=1)

def test_device(dev_name, diff_method):
    print(f"\nTesting {dev_name} ({diff_method})...")
    dev = qml.device(dev_name, wires=N)
    
    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(x):
        qml.AmplitudeEmbedding(features=x, wires=range(N), normalize=True)
        qml.RY(0.5, wires=0)
        return [qml.expval(qml.PauliZ(i)) for i in range(N)]
    
    # Warmup
    _ = circuit(inputs[0])
    
    t0 = time.time()
    res = torch.stack([circuit(x) for x in inputs])
    t1 = time.time()
    print(f"  Forward: {t1-t0:.4f}s")
    
    loss = res.sum()
    t0 = time.time()
    loss.backward()
    t1 = time.time()
    print(f"  Backward: {t1-t0:.4f}s")

for dev, diff in [("default.qubit", "backprop"), ("lightning.qubit", "adjoint"), ("lightning.gpu", "adjoint")]:
    try:
        test_device(dev, diff)
    except Exception as e:
        print(f"Error with {dev}: {e}")
