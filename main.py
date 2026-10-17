import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms


torch.manual_seed(60)
np.random.seed(60)

if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")

print("Kullanilan cihaz:", device)


# Veri setinin yüklenmesi, bölütlenmesi ve DataLoader'ların hazırlanması
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081))])

full_train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)


test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)



train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size


generator = torch.Generator().manual_seed(60)
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator = generator)


train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size= 64, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)


# MLP model mimarisinin tanımlanması
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(28*28, 256)

        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Modelin oluşturulması, kayıp fonksiyonu ve optimizer tanımları
model = MLP().to(device)

print("Model: ", model)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr = 0.001)

NUM_EPOCHS = 10

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

starting_time = time.time()

# Model eğitimi ve doğrulama döngüsü
for epoch in range(NUM_EPOCHS):
    model.train()

    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()


    epoch_train_loss = running_train_loss / total_train
    epoch_train_acc = (correct_train / total_train) * 100



    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()


        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)

        print("Epoch: ", epoch + 1)
        print("Train Loss: ", epoch_train_loss)
        print("Train Accuracy: ", epoch_train_acc)
        print("Validation Loss: ", epoch_val_loss)
        print("Validation Accuracy: ", epoch_val_acc)

        
finishing_time = time.time()
training_time = finishing_time - starting_time
print(f"Egitim suresi: %{training_time:.2f} saniye")



# Test kümesi üzerinde modelin değerlendirilmesi
model.eval()
correct_test = 0
total_test = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()


test_accuacy = (correct_test / total_test) * 100
print("Test basarisi: ", test_accuacy)



# Eğitim ve doğrulama süreç grafiklerinin çizilmesi ve kaydedilmesi
epoch_range = range(1, NUM_EPOCHS + 1)


plt.figure(figsize=(12, 5))

plt.subplot(1,2,1)
plt.plot(epoch_range, train_losses, label="Train Loss", color="blue")
plt.plot(epoch_range, val_losses, label = "Validation Loss", color = "red", marker="o")

plt.title("Eğitim ve Doğrulama Hatası")

plt.xlabel("Epoch")
plt.ylabel("Loss")


plt.grid(True)
plt.legend()



plt.subplot(1,2,2)
plt.plot(epoch_range, train_accuracies, label="Train Accuracy", color="green")
plt.plot(epoch_range, val_accuracies, label = "Validation Accuracy", color = "orange", marker="o")

plt.title("Eğitim ve Doğrulama Başarısı")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")


plt.grid(True)
plt.legend()


plt.tight_layout()
plt.savefig("mnist_deep_mlp.png", dpi=300)

plt.show()


# Kendi el yazısı verilerinin ön işlenmesi ve model testi
def process_image(image_path):
    img = Image.open(image_path).convert("L")
    arr = 255.0 - np.array(img, dtype = np.float32)
    arr = arr - arr.min()
    threshold = 0.30 * arr.max()
    arr = (arr / arr.max()) * 255.0

    clean_img = Image.fromarray(arr.astype(np.uint8))
    clean_img = clean_img.resize((28, 28), Image.Resampling.LANCZOS)

    tensor_img = transform(clean_img).unsqueeze(0).to(device)
    
    return tensor_img, clean_img


image_names = [
    "image.png",
    "image copy.png",
    "image copy 2.png",
    "image copy 3.png",
    "image copy 4.png",
    "image copy 5.png",
    "image copy 6.png",
    "image copy 7.png",
    "image copy 8.png",
    "image copy 9.png",
    "image copy 10.png"
]

real_digits = [2, 4, 1, 2, 0, 2, 0, 5, 0, 0, 2]

print("El yazi tahmini")

model.eval()
predictions = []
confidences = []
preprocessed_images = []

with torch.no_grad():
    for i, file_name in enumerate(image_names):
        img_path = os.path.join("rakamlar", file_name)
        tensor_img, clean_img = process_image(img_path)
        
        outputs = model(tensor_img)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        pred_digit = predicted.item()
        conf_score = confidence.item() * 100

        predictions.append(pred_digit)
        confidences.append(conf_score)
        preprocessed_images.append(clean_img)


        print(f"Tahmin edilen rakam: {pred_digit}, Gercek rakam: {real_digits[i]}, Guven skoru: {conf_score}")

correct_predict_count = 0

for p, a in zip(predictions, real_digits):
    if p == a:
        correct_predict_count += 1

accuracy = (correct_predict_count / len(real_digits)) * 100

print("El yazi tahmin basarisi: ", correct_predict_count / len(real_digits))



# Tahmin sonuçlarının görselleştirilmesi ve kaydedilmesi
plt.figure(figsize=(15, 3.5))

for i in range(len(image_names)):
    plt.subplot(1, 11, i + 1)
    plt.imshow(preprocessed_images[i], cmap='gray')

    is_correct = (predictions[i] == real_digits[i])
    title_color = "green" if is_correct else "red"

    plt.title(f"Gerçek: {real_digits[i]}\nTahmin: {predictions[i]}\n%{confidences[i]:.0f}", color=title_color, fontsize=10)
    plt.axis('off')

plt.suptitle(f"Doğruluk: {correct_predict_count}/{len(real_digits)} (%{accuracy:.1f})", fontsize=13)

plt.tight_layout()

plt.savefig("kendi_rakamlarim_tahmin.png", dpi=300)

plt.show()

        
    
    


