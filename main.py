import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define a arquitetura da rede neural
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define os parâmetros da rede neural
input_size = 784  # 28x28 pixels
hidden_size = 500
num_classes = 10  # 0-9 dígitos

# Carrega o conjunto de dados MNIST e aplica transformações
train_dataset = datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='data/', train=False, transform=transforms.ToTensor())

# Divide o conjunto de treinamento em lotes (batches)
batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Instancia a rede neural
model = NeuralNet(input_size, hidden_size, num_classes)

# Define a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento da rede neural
num_epochs = 5
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.reshape(-1, 28*28)  # Achatando as imagens em vetores de 784 dimensões
        targets = targets

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass e otimização
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 100 == 0:
            print(f'Epoca {epoch+1}/{num_epochs}, Lote {batch_idx+1}/{len(train_loader)}, Perda: {loss.item():.4f}')

# Avaliação da rede neural
with torch.no_grad():
    correct = 0
    total = 0
    for data, targets in test_dataset:
        data = data.reshape(-1, 28*28)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += 1
        correct += (predicted == targets).sum().item()

    accuracy = correct / total
    print(f'Acuracia da rede neural nos dados de teste: {accuracy:.4f}')

# Função para visualizar um exemplo de imagem
def visualize_image(image, prediction):
    plt.imshow(image.view(28, 28), cmap='gray')
    plt.title(f'Previsão: {prediction}')
    plt.show()

# Visualização de alguns exemplos
num_examples = 5
examples = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=num_examples, shuffle=True)
for example_batch, target_batch in examples:
    example_batch = example_batch.reshape(-1, 28*28)
    output_batch = model(example_batch)
    _, predicted_batch = torch.max(output_batch.data, 1)

    for i in range(num_examples):
        visualize_image(example_batch[i], predicted_batch[i])
