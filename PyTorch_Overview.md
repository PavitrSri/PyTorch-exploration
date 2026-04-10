# PyTorch - Package/Library Overview

## 1. Which package/library did you select?

**PyTorch** (`torch`), an open-source deep learning framework built by Meta AI Research [1].

The program also pulls in a few supporting libraries:
- `torch.nn` for the actual network layers
- `torch.optim` for updating weights during training
- `torchvision` for grabbing the MNIST dataset
- `matplotlib` for saving a visual of the results

---

## 2. What is the package/library?

### What purpose does it serve?

At its core, PyTorch does two things: it lets you work with tensors and it automatically computes gradients, which is what makes training a neural network possible [1]. Without gradients, the model has no way of knowing which direction to adjust its weights.


### How do you use it?

The program starts simple by picking a device and defining a transform:

```python
device = torch.device("cpu")

transform = transforms.ToTensor()
```

`transforms.ToTensor()` converts each image into a tensor and scales pixel values from 0–255 down to 0–1, which helps the model train more stably [2].

Then the data gets loaded. We can download an entire dataset in just a few lines:

```python
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=64)
```

`DataLoader` handles splitting the 60,000 images into batches of 64 and shuffling them each epoch so the model doesn't accidentally memorize the order [6].

The model itself is a Python class:

```python
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
```

Then a loss function and optimizer are set up:

```python
loss_fn   = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

And training runs for 5 epochs:

```python
for epoch in range(5):
    correct = 0
    total   = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss    = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted  = outputs.argmax(dim=1)
        correct   += (predicted == labels).sum().item()
        total     += labels.size(0)

    print(f"Epoch {epoch+1} Accuracy: {(correct/total)*100:.2f}%")
```

After training, the model is tested on data it has never seen [1]:

```python
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs        = model(images)
        predicted      = outputs.argmax(dim=1)
        correct       += (predicted == labels).sum().item()
        total         += labels.size(0)

print("\nTest Accuracy:", 100 * correct / total, "%")
```

Finally, the raw outputs are turned into confidence scores and matplotlib saves the visual [3]:

```python
model.eval()
sample_images, sample_labels = next(iter(test_loader))
sample_images = sample_images[:10].to(device)

with torch.no_grad():
    outputs           = model(sample_images)
    probs             = torch.softmax(outputs, dim=1)
    predicted_classes = torch.argmax(probs, dim=1)

for i in range(10):
    pred = predicted_classes[i].item()
    conf = probs[i, pred].item()
    print(f"Image {i}: {pred} | Confidence: {conf:.2%}")

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    img = sample_images[i][0].cpu().numpy()
    plt.imshow(img, cmap='gray')
    plt.title(f"Pred: {predicted_classes[i].item()}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("my_results.png")
```

---


## 3. When was it created?

PyTorch was released by **Meta AI Research** in **January 2017** [4]. Before it came along, most deep learning frameworks required you to define your entire computation graph upfront before running anything which made debugging a nightmare. PyTorch changed that by building the graph dynamically as code runs, so you could use regular Python print statements and debuggers to figure out what was going wrong. That alone was enough to win over most of the research community [4].

---

## 5. Why did you select this package/library?

I picked PyTorch because I wanted to learn something that would actually be useful outside of this course. It's the framework behind most of the machine learning research being published right now, and it's used in production at companies like Meta, Tesla, and OpenAI [5].

---

## 6. How did learning this package/library influence your learning of the language?

Working through PyTorch made a few Python concepts click in ways they hadn't before.

Subclassing `nn.Module` showed importance of inheritance, if you don't call `super().__init__()`, your layers won't register properly and the model breaks [1].

`with torch.no_grad():` was also a good reminder that `with` statements aren't just for opening files. And indexing into tensors with `probs[i, pred]` or `sample_images[i][0]` pushed me to think more carefully about how multi-dimensional data is structured, which is different from regular Python lists [1].

---

## 7. Overall experience with the package/library

### When would you recommend it?

I'd recommend PyTorch to anyone who is curious about machine learning and wants to build something real with it. It is beginner-friendly enough to get started quickly, but deep enough that you won't outgrow it. Whether you're a student exploring neural networks for the first time or someone building a project from scratch, PyTorch gives you the tools to do it without getting in your way [4].

### Would you continue using it? Why or why not?

Yes. The fundamentals learned here like writing a model class, running a training loop, using softmax for confidence scores, carry over directly to more advanced projects. Getting comfortable with PyTorch at this level makes it accessible to take on more complex projects in machine learning [1].

---

## References

[1] PyTorch. *PyTorch Documentation*. https://pytorch.org/docs/stable/index.html

[2] torchvision. *torchvision.transforms Documentation*. https://pytorch.org/vision/stable/transforms.html

[3] Matplotlib. *Matplotlib Documentation*. https://matplotlib.org/stable/index.html

[4] PyTorch. *PyTorch — Getting Started*. https://pytorch.org/get-started/locally/

[5] PyTorch. *PyTorch — Training a Classifier Tutorial*. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

[6] GeeksforGeeks. *MNIST Dataset in Python*. https://www.geeksforgeeks.org/mnist-dataset/

---

