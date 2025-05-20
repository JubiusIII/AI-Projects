import pygame
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from NNFromScratch import modelMNIST, device  

WIDTH, HEIGHT = 280, 280
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RADIUS = 20

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw a digit (Press S to predict, Enter to clear)")
clock = pygame.time.Clock()
screen.fill(WHITE)

def show_resized_image(image_path):
    img = Image.open(image_path).convert('L')
    img = Image.fromarray(255 - np.array(img))

    arr = np.array(img)
    non_empty_columns = np.where(arr.min(axis=0) < 255)[0]
    non_empty_rows = np.where(arr.min(axis=1) < 255)[0]
    if non_empty_columns.size and non_empty_rows.size:
        crop_box = (
            non_empty_columns[0],
            non_empty_rows[0],
            non_empty_columns[-1] + 1,
            non_empty_rows[-1] + 1
        )
        img = img.crop(crop_box)

    img = img.resize((28, 28), Image.LANCZOS)
    img.save("digit_converted.png", format="PNG")
    return img

def predict_and_learn(image_path):
    img = show_resized_image(image_path)
    img_tensor = torch.tensor(np.array(img).flatten(), dtype=torch.float32, device=device).unsqueeze(0) / 255.0

    # Forward pass
    output = modelMNIST.forward(img_tensor)
    predicted_label = torch.argmax(output, dim=1).item()

    print(f"Predicted Label: {predicted_label}")

    # Ask user for correct label
    try:
        correct_label = int(input("What was the correct digit? (0-9): "))
        if not 0 <= correct_label <= 9:
            raise ValueError("Must be between 0 and 9")
    except Exception as e:
        print(f"Invalid input: {e}")
        return

    # Backward pass
    target = torch.tensor([correct_label], dtype=torch.long, device=device)
    loss = F.cross_entropy(output, target)

    # Zero grads, backprop, and update weights
    for w in modelMNIST.weights:
        w.requires_grad = True
    for b in modelMNIST.biases:
        b.requires_grad = True

    loss.backward()
    learning_rate = 0.01

    # Manually update weights (no optimizer)
    with torch.no_grad():
        for i in range(len(modelMNIST.weights)):
            modelMNIST.weights[i] -= learning_rate * modelMNIST.weights[i].grad
            modelMNIST.biases[i] -= learning_rate * modelMNIST.biases[i].grad
            modelMNIST.weights[i].grad.zero_()
            modelMNIST.biases[i].grad.zero_()

    print("Model updated using backpropagation.")

drawing = False
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if pygame.mouse.get_pressed()[0]:
            x, y = pygame.mouse.get_pos()
            pygame.draw.circle(screen, BLACK, (x, y), RADIUS)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                screen.fill(WHITE)

            if event.key == pygame.K_s:
                pygame.image.save(screen, "digit.png")
                print("Saved drawing as 'digit.png'")
                predict_and_learn("digit.png")

            if event.key == pygame.K_RETURN:
                screen.fill(WHITE)
                print("Canvas cleared.")

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
