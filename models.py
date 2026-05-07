import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class ClassicalCNN(nn.Module):
    def __init__(self, num_classes=115):
        super(ClassicalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=2)
        self.conv2 = nn.Conv2d(8, 5, kernel_size=2)
        self.conv3 = nn.Conv2d(5, 3, kernel_size=2)
        self.pool = nn.AvgPool2d(2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(3, num_classes)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = x.view(-1, 1, 16, 16)
        x = self.pool(self.tanh(self.conv1(x)))
        x = self.pool(self.tanh(self.conv2(x)))
        x = self.pool(self.tanh(self.conv3(x)))
        x = torch.mean(x, dim=(2, 3))
        x = self.dropout(x)
        x = self.fc(x)
        return x

class TrainableQuanvolution(nn.Module):
    def __init__(self):
        super(TrainableQuanvolution, self).__init__()
        try:
            self.dev = qml.device("lightning.gpu", wires=4)
        except Exception:
            self.dev = qml.device("lightning.qubit", wires=4)
            
        weight_shapes = {"weights": (1, 4, 3)}
        
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(4))
            qml.StronglyEntanglingLayers(weights, wires=range(4))
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]
            
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, 16, 16)
        x_unfold = F.unfold(x, kernel_size=2, stride=2)
        x_patches = x_unfold.transpose(1, 2).reshape(-1, 4)
        
        q_results = self.qlayer(x_patches)
        
        q_results = q_results.view(batch_size, 64, 4).transpose(1, 2)
        out = q_results.reshape(batch_size, 4, 8, 8)
        return out

class HybridQNN(nn.Module):
    def __init__(self, num_classes=115):
        super(HybridQNN, self).__init__()
        self.qconv = TrainableQuanvolution()
        self.fc = nn.Linear(4 * 8 * 8, num_classes)
        
    def forward(self, x):
        x = self.qconv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class MultiClassQCNN(nn.Module):
    def __init__(self, num_classes=115, num_layers=6):
        super(MultiClassQCNN, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        try:
            self.dev = qml.device("lightning.gpu", wires=8)
        except Exception:
            self.dev = qml.device("lightning.qubit", wires=8)
        
        weight_shapes = {
            "f1_weights": (num_layers, 8, 2),
            "f2_weights": (num_layers, 8, 15),
            "pool_weights": (2,)
        }
        
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, f1_weights, f2_weights, pool_weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(8), normalize=True)
            
            for layer in range(self.num_layers):
                for i in range(8):
                    qml.RY(f1_weights[layer, i, 0], wires=i)
                    qml.RX(f1_weights[layer, i, 1], wires=i)
                
                for i in range(8):
                    w1 = i
                    w2 = (i + 1) % 8
                    qml.ArbitraryUnitary(f2_weights[layer, i], wires=[w1, w2])
            
            qml.CRZ(pool_weights[0], wires=[0, 1])
            qml.CRX(pool_weights[1], wires=[0, 1])
            
            return qml.probs(wires=range(1, 8))
            
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        
    def forward(self, x):
        probs = self.qlayer(x)
        return probs[:, :self.num_classes]
