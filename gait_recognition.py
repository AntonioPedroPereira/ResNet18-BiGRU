# Part 1: Imports and Logging Setup
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights
from tqdm import tqdm
import logging
from collections import Counter

logging.basicConfig(
    filename='gait_recognition.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Part 2: Custom Dataset Class (GaitDataset)
class GaitDataset(Dataset):
    def __init__(self, sequence_dirs, labels, max_frames=30, min_valid_frames=5):
        self.sequence_dirs = sequence_dirs
        self.labels = labels
        self.max_frames = max_frames
        self.min_valid_frames = min_valid_frames
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((64, 44)),
            T.Normalize(mean=[0.2], std=[0.3]),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10)
        ])

    def __len__(self):
        return len(self.sequence_dirs)

    def __getitem__(self, idx):
        sequence_dir = self.sequence_dirs[idx]
        try:
            image_files = sorted([f for f in os.listdir(sequence_dir) if f.endswith('.png')])
        except Exception as e:
            logging.warning(f"Error accessing {sequence_dir}: {str(e)}")
            return torch.zeros(self.max_frames, 1, 64, 44), torch.tensor(self.labels[idx], dtype=torch.long)

        silhouettes = []
        for img_file in image_files[:self.max_frames]:
            img_path = os.path.join(sequence_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None or img.mean() < 5:
                continue
            img = self.transform(img).float()
            silhouettes.append(img)

        if len(silhouettes) < self.min_valid_frames:
            logging.warning(f"Insufficient valid images ({len(silhouettes)}) in {sequence_dir}")
            return torch.zeros(self.max_frames, 1, 64, 44), torch.tensor(self.labels[idx], dtype=torch.long)

        try:
            silhouettes = torch.stack(silhouettes)
            if len(silhouettes) < self.max_frames:
                padding = torch.zeros(self.max_frames - len(silhouettes), 1, 64, 44)
                silhouettes = torch.cat([silhouettes, padding], dim=0)
            else:
                silhouettes = silhouettes[:self.max_frames]
        except Exception as e:
            logging.warning(f"Error stacking images in {sequence_dir}: {str(e)}")
            return torch.zeros(self.max_frames, 1, 64, 44), torch.tensor(self.labels[idx], dtype=torch.long)

        return silhouettes, torch.tensor(self.labels[idx], dtype=torch.long)

# Part 3: Model Architecture (GaitRecognitionModel)
class GaitRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(GaitRecognitionModel, self).__init__()
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in self.cnn.parameters():
            param.requires_grad = True
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
        self.cnn.fc = nn.Linear(512, 128)
        self.gru = nn.GRU(input_size=128, hidden_size=256, num_layers=2, bidirectional=True,
                          batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256 * 2, num_classes)
        for m in self.cnn.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight.requires_grad:
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        for m in self.gru.modules():
            for param in m.parameters():
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)

    def forward(self, x):
        batch, T, C, H, W = x.size()
        x = x.view(batch * T, C, H, W)
        spatial_features = self.cnn(x)
        spatial_features = spatial_features.view(batch, T, 128)
        _, hidden = self.gru(spatial_features)
        temporal_features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        temporal_features = self.dropout(temporal_features)
        logits = self.fc(temporal_features)
        return logits, spatial_features

# Part 4: Dataset Scanning and Preprocessing
def main():
    data_dir = r"path_to_CASIAB" # Change to CASIA-B Path
    conditions = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06', 'bg-01', 'bg-02', 'cl-01', 'cl-02']
    viewpoints = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
    sequence_dirs = []
    labels = []
    min_valid_frames = 5

    print("Checking subject folders...")
    missing_subjects = []
    for subject_id in range(1, 125):
        subject_dir = os.path.join(data_dir, f"{subject_id:03d}")
        if not os.path.exists(subject_dir):
            missing_subjects.append(f"{subject_id:03d}")
            print(f"Missing subject folder: {subject_dir}")
            logging.debug(f"Missing subject folder: {subject_dir}")

    logging.debug(f"Found {len(missing_subjects)} missing subject folders: {', '.join(missing_subjects)}")
    if missing_subjects:
        print(f"Found {len(missing_subjects)} missing subject folders: {', '.join(missing_subjects)}")
    if len(missing_subjects) > 10:
        raise ValueError(f"Too many missing subject folders ({len(missing_subjects)}). Please verify the dataset.")

    print("Scanning dataset...")
    for subject_id in tqdm(range(1, 125), desc="Subjects"):
        subject_dir = os.path.join(data_dir, f"{subject_id:03d}")
        if not os.path.exists(subject_dir):
            logging.debug(f"Skipping subject {subject_id:03d}: directory not found")
            continue
        for condition in conditions:
            condition_dir = os.path.join(subject_dir, condition)
            if not os.path.exists(condition_dir):
                logging.debug(f"Condition folder {condition_dir} not found")
                continue
            for viewpoint in viewpoints:
                viewpoint_dir = os.path.join(condition_dir, viewpoint)
                if os.path.exists(viewpoint_dir):
                    try:
                        image_files = [f for f in os.listdir(viewpoint_dir) if f.endswith('.png')]
                        valid_images = 0
                        for img_file in image_files[:30]:
                            img_path = os.path.join(viewpoint_dir, img_file)
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None and img.mean() >= 5:
                                valid_images += 1
                        if valid_images >= min_valid_frames:
                            sequence_dirs.append(viewpoint_dir)
                            labels.append(subject_id - 1)
                        else:
                            logging.debug(f"Skipping sequence {viewpoint_dir} with {valid_images} valid images")
                    except Exception as e:
                        logging.warning(f"Error scanning {viewpoint_dir}: {str(e)}")
                        continue

    label_counts = Counter(labels)
    logging.debug(f"Label counts: {label_counts}")
    valid_labels = [label for label, count in label_counts.items() if count >= 2]
    filtered_dirs = []
    filtered_labels = []
    for dir_path, label in zip(sequence_dirs, labels):
        if label in valid_labels:
            filtered_dirs.append(dir_path)
            filtered_labels.append(label)

    if len(filtered_dirs) < 100:
        raise ValueError(
            f"Too few valid sequences ({len(filtered_dirs)}) after class filtering. Check dataset for missing or low-quality .png files.")

    print(f"Found {len(filtered_dirs)} valid sequences across {len(set(filtered_labels))} classes.")

    train_dirs, test_dirs, train_labels, test_labels = train_test_split(
        filtered_dirs, filtered_labels, test_size=0.3, stratify=filtered_labels, random_state=42
    )
    val_dirs, test_dirs, val_labels, test_labels = train_test_split(
        test_dirs, test_labels, test_size=0.5, stratify=test_labels, random_state=42
    )

    train_dataset = GaitDataset(train_dirs, train_labels, max_frames=30, min_valid_frames=5)
    val_dataset = GaitDataset(val_dirs, val_labels, max_frames=30, min_valid_frames=5)
    test_dataset = GaitDataset(test_dirs, test_labels, max_frames=30, min_valid_frames=5)
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=2)

    # Part 5: Training and Evaluation Loops
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    num_classes = len(set(filtered_labels))
    model = GaitRecognitionModel(num_classes=num_classes).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0003, betas=(0.9, 0.999),
                           weight_decay=5e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion_ce = nn.CrossEntropyLoss()

    num_epochs = 50
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        valid_batches = 0
        correct_train = 0
        total_train = 0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training") as pbar:
            for data, target in pbar:
                if data.sum() == 0:
                    logging.warning(f"Skipping batch with empty sequence")
                    continue
                try:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    logits, _ = model(data)
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logging.warning(f"NaN/Inf in logits at epoch {epoch + 1}")
                        continue
                    loss = criterion_ce(logits, target)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    valid_batches += 1
                    pred = logits.argmax(dim=1)
                    correct_train += pred.eq(target).sum().item()
                    total_train += target.size(0)
                    train_accuracy = 100. * correct_train / total_train if total_train > 0 else 0
                    pbar.set_postfix(loss=running_loss / valid_batches, accuracy=train_accuracy)
                except RuntimeError as e:
                    logging.warning(f"Training error at epoch {epoch + 1}: {str(e)}")
                    torch.cuda.empty_cache()
                    continue
        scheduler.step()
        train_accuracy = 100. * correct_train / total_train if total_train > 0 else 0
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / valid_batches:.4f}, Training Accuracy: {train_accuracy:.2f}%")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation") as pbar:
                for data, target in pbar:
                    if data.sum() == 0:
                        continue
                    try:
                        data, target = data.to(device), target.to(device)
                        logits, _ = model(data)
                        if torch.isnan(logits).any() or torch.isinf(logits).any():
                            logging.warning(f"NaN/Inf in validation logits at epoch {epoch + 1}")
                            continue
                        pred = logits.argmax(dim=1)
                        correct += pred.eq(target).sum().item()
                        total += target.size(0)
                        pbar.set_postfix(accuracy=100. * correct / total if total > 0 else 0)
                    except RuntimeError as e:
                        logging.warning(f"Validation error at epoch {epoch + 1}: {str(e)}")
                        torch.cuda.empty_cache()
                        continue
        val_accuracy = 100. * correct / total if total > 0 else 0
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_gait_model.pth')
            print(f"Saved best model with validation accuracy: {val_accuracy:.2f}%")

    # Save the final model
    torch.save(model.state_dict(), 'final_gait_model.pth')
    print("Saved final model as 'final_gait_model.pth'")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing") as pbar:
            for data, target in pbar:
                if data.sum() == 0:
                    continue
                try:
                    data, target = data.to(device), target.to(device)
                    logits, _ = model(data)
                    pred = logits.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                    pbar.set_postfix(accuracy=100. * correct / total if total > 0 else 0)
                except RuntimeError as e:
                    logging.warning(f"Testing error: {str(e)}")
                    torch.cuda.empty_cache()
                    continue
    test_accuracy = 100. * correct / total if total > 0 else 0
    print(f"Test Accuracy: {test_accuracy:.2f}%")

# Part 6: Video/Camera Preprocessing and Inference
def preprocess_video_or_camera(input_source, max_frames=30, output_dir='temp_frames'):
    """
    Preprocess an MP4 video or camera feed into silhouette frames.
    input_source: Path to MP4 file or 0 for camera
    max_frames: Maximum number of frames to process
    output_dir: Directory to save temporary frames
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        logging.error(f"Could not open input source: {input_source}")
        return None

    frames = []
    frame_count = 0
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((64, 44)),
        T.Normalize(mean=[0.2], std=[0.3])
    ])

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale and apply simple thresholding for silhouette
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, silhouette = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        silhouette = silhouette.astype(np.uint8)
        # Save frame temporarily for verification
        cv2.imwrite(os.path.join(output_dir, f'frame_{frame_count:04d}.png'), silhouette)
        silhouette_tensor = transform(silhouette).float()
        frames.append(silhouette_tensor)
        frame_count += 1

    cap.release()
    if len(frames) < 5:
        logging.warning(f"Insufficient valid frames ({len(frames)}) from {input_source}")
        return None

    # Stack frames and pad if necessary
    frames = torch.stack(frames)
    if len(frames) < max_frames:
        padding = torch.zeros(max_frames - len(frames), 1, 64, 44)
        frames = torch.cat([frames, padding], dim=0)
    else:
        frames = frames[:max_frames]

    return frames.unsqueeze(0)  # Add batch dimension

def infer_gait(model, input_tensor, device, num_classes):
    """
    Perform gait recognition inference on preprocessed input tensor.
    """
    model.eval()
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        logits, _ = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()
        confidence = probabilities[0, pred_class].item()
    return pred_class, confidence

def process_and_infer(input_source, model_path='final_gait_model.pth', num_classes=124, max_frames=30):
    """
    Load model and perform inference on video or camera input.
    input_source: Path to MP4 file or 0 for camera
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaitRecognitionModel(num_classes=num_classes).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None, None

    input_tensor = preprocess_video_or_camera(input_source, max_frames=max_frames)
    if input_tensor is None:
        print("Failed to preprocess input.")
        return None, None

    pred_class, confidence = infer_gait(model, input_tensor, device, num_classes)
    return pred_class, confidence

if __name__ == "__main__":
    print("Gait Recognition System")
    print("Please select an action:")
    print("1. Train the model")
    print("2. Test on an MP4 video")
    print("3. Test on camera")
    try:
        choice = int(input("Enter your choice (1, 2, or 3): "))
    except ValueError:
        print("Invalid input. Please enter 1, 2, or 3.")
        exit(1)

    if choice == 1:
        print("Starting model training...")
        main()  # Run the training pipeline
    elif choice == 2:
        video_path = r"Video_Path"  #Add video Path
        print(f"Processing video: {video_path}")
        pred_class, confidence = process_and_infer(video_path, model_path='final_gait_model.pth', num_classes=124)
        if pred_class is not None:
            print(f"MP4 Video - Predicted class: {pred_class}, Confidence: {confidence:.4f}")
        else:
            print("MP4 Video - Inference failed. Check gait_recognition.log for details.")
    elif choice == 3:
        print("Starting camera inference (press Ctrl+C to stop)...")
        pred_class, confidence = process_and_infer(0, model_path='final_gait_model.pth', num_classes=124)
        if pred_class is not None:
            print(f"Camera - Predicted class: {pred_class}, Confidence: {confidence:.4f}")
        else:
            print("Camera - Inference failed. Check gait_recognition.log for details.")
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
        exit(1)