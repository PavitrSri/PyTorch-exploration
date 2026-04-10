import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


device = torch.device("cpu")


transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

print("Training samples:", len(train_data))
print("Test samples:", len(test_data))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100) 
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Model().to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


print("\nTRAINING...\n")

for epoch in range(5):
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1} Accuracy: {(correct/total)*100:.2f}%")


correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        predicted = outputs.argmax(dim=1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print("\nTest Accuracy:", 100 * correct / total, "%")



model.eval()
sample_images, sample_labels = next(iter(test_loader))
sample_images = sample_images[:10].to(device)

with torch.no_grad():
    outputs = model(sample_images)
    probs = torch.softmax(outputs, dim=1)
    predicted_classes = torch.argmax(probs, dim=1)

for i in range(10):
    pred = predicted_classes[i].item()
    conf = probs[i, pred].item()





    print(f"Image {i}: {pred} | Confidence: {conf:.2%}")


# =========================
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    # just grab the image from the tensor
    img = sample_images[i][0].cpu().numpy() 
    plt.imshow(img, cmap='gray')
    plt.title(f"Pred: {predicted_classes[i].item()}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("my_results.png")
print("Saved results to my_results.png")