# Script 2: Model Training and Prediction

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.optim import Adam

# Define the GCN model
class GCNClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Load graph data
pyg_data = torch.load("graph_data.pt")

# Prepare the dataset
num_classes = 2  # Functional or Non-Functional
pyg_data.y = torch.randint(0, num_classes, (pyg_data.num_nodes,))  # Example random labels
pyg_data.train_mask = torch.rand(pyg_data.num_nodes) < 0.8
pyg_data.test_mask = ~pyg_data.train_mask

# Define model and training components
model = GCNClassifier(input_dim=pyg_data.num_node_features, hidden_dim=64, output_dim=num_classes)
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(pyg_data)
    loss = criterion(out[pyg_data.train_mask], pyg_data.y[pyg_data.train_mask])
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
_, pred = model(pyg_data).max(dim=1)
correct = (pred[pyg_data.test_mask] == pyg_data.y[pyg_data.test_mask]).sum()
accuracy = int(correct) / int(pyg_data.test_mask.sum())
print(f"Test Accuracy: {accuracy:.4f}")

# Prediction for a new UserStory
new_user_story = {"user_story": "As a healthcare administrator, I want the system to ensure patient data is encrypted during storage and transmission so that it complies with HIPAA regulations and ensures data security."}
new_user_feature = torch.tensor([hash(new_user_story["user_story"]) % 1000], dtype=torch.float)
pyg_data.x = torch.cat([pyg_data.x, new_user_feature.unsqueeze(0)], dim=0)

# Predict the RequirementType
model.eval()
prediction = model(pyg_data)
new_node_pred = prediction[-1].argmax().item()
print("Predicted RequirementType:", "Functional" if new_node_pred == 1 else "Non-Functional")
