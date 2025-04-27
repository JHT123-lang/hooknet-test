# import torch
# from torch.utils.data import DataLoader
# import pdb

# def train_model(model, train_loader, val_loader, criterion, dice_loss, optimizer, num_epochs, device, lambda_weight=0.7):
#     model.to(device)
#     best_val_loss = float("inf")
#     for epoch in range(num_epochs):
#         print(f"开始第{epoch}个epoch")
#         model.train()
#         train_loss = 0.0
#         train_correct = 0
#         train_total = 0
#         for context, target, masks in train_loader:
#             context, target, masks = context.to(device), target.to(device), masks.to(device).long()
#             optimizer.zero_grad()
#             outputs = model(context, target)
#             # print(outputs['target_out'].shape)
#             # print(masks.shape)
#             # pdb.set_trace()
#             c_out, t_out = outputs["context_out"], outputs["target_out"]
#             ce_loss = lambda_weight * criterion(t_out, masks) + (1 - lambda_weight) * criterion(c_out, masks)
#             dice = dice_loss(t_out, masks)
#             loss = 0.5 * ce_loss + 0.5 * dice
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * target.size(0)

#             # 计算训练准确率
#             _, predicted = torch.max(t_out, 1)
#             train_total += masks.numel()
#             train_correct += (predicted == masks).sum().item()

#         train_loss /= len(train_loader.dataset)
#         train_acc = train_correct / train_total
#         print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

#         model.eval()
#         val_loss = 0.0
#         val_correct = 0
#         val_total = 0
#         with torch.no_grad():
#             for context, target, masks in val_loader:
#                 context, target, masks = context.to(device), target.to(device), masks.to(device).long()
#                 outputs = model(context, target)
#                 c_out, t_out = outputs["context_out"], outputs["target_out"]
#                 ce_loss = lambda_weight * criterion(t_out, masks) + (1 - lambda_weight) * criterion(c_out, masks)
#                 dice = dice_loss(t_out, masks)
#                 loss = 0.5 * ce_loss + 0.5 * dice
#                 val_loss += loss.item() * target.size(0)

#                 # 计算验证准确率
#                 _, predicted = torch.max(t_out, 1)
#                 val_total += masks.numel()
#                 val_correct += (predicted == masks).sum().item()

#         val_loss /= len(val_loader.dataset)
#         val_acc = val_correct / val_total
#         print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), "best_hooknet.pth")



# import torch
# from torch.utils.data import DataLoader
# import numpy as np
# from tabulate import tabulate
# import pdb

# def calculate_metrics(pred, target, num_classes):
#     """
#     Calculate IoU and Accuracy per class.
#     pred: tensor of shape (batch, height, width) with predicted class indices
#     target: tensor of shape (batch, height, width) with ground truth class indices
#     num_classes: number of classes
#     """
#     iou_per_class = []
#     acc_per_class = []
    
#     for cls in range(num_classes):
#         pred_cls = (pred == cls)
#         target_cls = (target == cls)
        
#         # Intersection and Union for IoU
#         intersection = (pred_cls & target_cls).float().sum()
#         union = (pred_cls | target_cls).float().sum()
#         iou = intersection / (union + 1e-6) * 100  # Convert to percentage
        
#         # Accuracy per class
#         correct = (pred_cls & target_cls).float().sum()
#         total = target_cls.float().sum()
#         acc = correct / (total + 1e-6) * 100  # Convert to percentage
        
#         iou_per_class.append(iou.item())
#         acc_per_class.append(acc.item())
    
#     return iou_per_class, acc_per_class

# def train_model(model, train_loader, val_loader, criterion, dice_loss, optimizer, num_epochs, device, lambda_weight=0.7):
#     model.to(device)
#     best_val_loss = float("inf")
#     num_classes = 4  # Assuming 4 classes: background, AGG, FOL-I, FOL-II
#     class_names = ["background", "AGG", "FOL-I", "FOL-II"]
    
#     for epoch in range(num_epochs):
#         print(f"开始第{epoch}个epoch")
#         model.train()
#         train_loss = 0.0
#         train_correct = 0
#         train_total = 0
#         train_iou = np.zeros(num_classes)
#         train_acc = np.zeros(num_classes)
#         train_batches = 0
        
#         for context, target, masks in train_loader:
#             context, target, masks = context.to(device), target.to(device), masks.to(device).long()
#             optimizer.zero_grad()
#             # print(context.shape,target.shape,masks.shape)

#             outputs = model(context, target)
#             c_out, t_out = outputs["context_out"], outputs["target_out"]
#             # print(c_out.shape, t_out.shape)
#             # pdb.set_trace()
#             ce_loss = lambda_weight * criterion(t_out, masks) + (1 - lambda_weight) * criterion(c_out, masks)
#             dice = dice_loss(t_out, masks)
#             loss = 0.5 * ce_loss + 0.5 * dice
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * target.size(0)

#             # Calculate metrics
#             _, predicted = torch.max(t_out, 1)
#             train_total += masks.numel()
#             train_correct += (predicted == masks).sum().item()
            
#             # Calculate IoU and Accuracy per class
#             iou, acc = calculate_metrics(predicted, masks, num_classes)
#             train_iou += np.array(iou)
#             train_acc += np.array(acc)
#             train_batches += 1

#         train_loss /= len(train_loader.dataset)
#         train_acc_global = train_correct / train_total
#         train_iou /= train_batches
#         train_acc /= train_batches
        
#         # Prepare and print training metrics table
#         train_table = [[class_names[i], f"{train_iou[i]:.2f}", f"{train_acc[i]:.2f}"] for i in range(num_classes)]
#         print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc_global:.4f}")
#         print("Training Metrics:")
#         print(tabulate(train_table, headers=["Class", "IoU", "Acc"], tablefmt="grid"))

#         model.eval()
#         val_loss = 0.0
#         val_correct = 0
#         val_total = 0
#         val_iou = np.zeros(num_classes)
#         val_acc = np.zeros(num_classes)
#         val_batches = 0
        
#         with torch.no_grad():
#             for context, target, masks in val_loader:
#                 context, target, masks = context.to(device), target.to(device), masks.to(device).long()
#                 outputs = model(context, target)
#                 c_out, t_out = outputs["context_out"], outputs["target_out"]
#                 ce_loss = lambda_weight * criterion(t_out, masks) + (1 - lambda_weight) * criterion(c_out, masks)
#                 dice = dice_loss(t_out, masks)
#                 loss = 0.5 * ce_loss + 0.5 * dice
#                 val_loss += loss.item() * target.size(0)

#                 # Calculate metrics
#                 _, predicted = torch.max(t_out, 1)
#                 val_total += masks.numel()
#                 val_correct += (predicted == masks).sum().item()
                
#                 # Calculate IoU and Accuracy per class
#                 iou, acc = calculate_metrics(predicted, masks, num_classes)
#                 val_iou += np.array(iou)
#                 val_acc += np.array(acc)
#                 val_batches += 1

#         val_loss /= len(val_loader.dataset)
#         val_acc_global = val_correct / val_total
#         val_iou /= val_batches
#         val_acc /= val_batches
        
#         # Prepare and print validation metrics table
#         val_table = [[class_names[i], f"{val_iou[i]:.2f}", f"{val_acc[i]:.2f}"] for i in range(num_classes)]
#         print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc_global:.4f}")
#         print("Validation Metrics:")
#         print(tabulate(val_table, headers=["Class", "IoU", "Acc"], tablefmt="grid"))
        
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), "best_hooknet.pth")


import torch
from torch.utils.data import DataLoader
import numpy as np
from tabulate import tabulate
import logging

def calculate_metrics(pred, target, num_classes):
    """
    Calculate IoU and Accuracy per class.
    pred: tensor of shape (batch, height, width) with predicted class indices
    target: tensor of shape (batch, height, width) with ground truth class indices
    num_classes: number of classes
    """
    iou_per_class = []
    acc_per_class = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        # Intersection and Union for IoU
        intersection = (pred_cls & target_cls).float().sum()
        union = (pred_cls | target_cls).float().sum()
        iou = intersection / (union + 1e-6) * 100  # Convert to percentage
        
        # Accuracy per class
        correct = (pred_cls & target_cls).float().sum()
        total = target_cls.float().sum()
        acc = correct / (total + 1e-6) * 100  # Convert to percentage
        
        iou_per_class.append(iou.item())
        acc_per_class.append(acc.item())
    
    return iou_per_class, acc_per_class

def train_model(model, train_loader, val_loader, criterion, dice_loss, optimizer, num_epochs, device, lambda_weight=0.7):
    model.to(device)
    best_val_loss = float("inf")
    num_classes = 4  # Assuming 4 classes: background, AGG, FOL-I, FOL-II
    class_names = ["background", "AGG", "FOL-I", "FOL-II"]
    
    for epoch in range(num_epochs):
        print(f"开始第{epoch}个epoch")
        
        # 训练阶段
        if train_loader is not None:
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_iou = np.zeros(num_classes)
            train_acc = np.zeros(num_classes)
            train_batches = 0
            
            for context, target, masks in train_loader:
                context, target, masks = context.to(device), target.to(device), masks.to(device).long()
                optimizer.zero_grad()
                
                outputs = model(context, target)
                c_out, t_out = outputs["context_out"], outputs["target_out"]
                ce_loss = lambda_weight * criterion(t_out, masks) + (1 - lambda_weight) * criterion(c_out, masks)
                dice = dice_loss(t_out, masks)
                loss = 0.5 * ce_loss + 0.5 * dice
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * target.size(0)
                
                # Calculate metrics
                _, predicted = torch.max(t_out, 1)
                train_total += masks.numel()
                train_correct += (predicted == masks).sum().item()
                
                # Calculate IoU and Accuracy per class
                iou, acc = calculate_metrics(predicted, masks, num_classes)
                train_iou += np.array(iou)
                train_acc += np.array(acc)
                train_batches += 1
            
            train_loss /= len(train_loader.dataset)
            train_acc_global = train_correct / train_total
            train_iou /= train_batches
            train_acc /= train_batches
            
            # Prepare and print training metrics table
            train_table = [[class_names[i], f"{train_iou[i]:.2f}", f"{train_acc[i]:.2f}"] for i in range(num_classes)]
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc_global:.4f}")
            print("Training Metrics:")
            print(tabulate(train_table, headers=["Class", "IoU", "Acc"], tablefmt="grid"))
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, No training data provided")
        
        # 验证阶段
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_iou = np.zeros(num_classes)
            val_acc = np.zeros(num_classes)
            val_batches = 0
            
            with torch.no_grad():
                for context, target, masks in val_loader:
                    context, target, masks = context.to(device), target.to(device), masks.to(device).long()
                    outputs = model(context, target)
                    c_out, t_out = outputs["context_out"], outputs["target_out"]
                    ce_loss = lambda_weight * criterion(t_out, masks) + (1 - lambda_weight) * criterion(c_out, masks)
                    dice = dice_loss(t_out, masks)
                    loss = 0.5 * ce_loss + 0.5 * dice
                    val_loss += loss.item() * target.size(0)
                    
                    # Calculate metrics
                    _, predicted = torch.max(t_out, 1)
                    val_total += masks.numel()
                    val_correct += (predicted == masks).sum().item()
                    
                    # Calculate IoU and Accuracy per class
                    iou, acc = calculate_metrics(predicted, masks, num_classes)
                    val_iou += np.array(iou)
                    val_acc += np.array(acc)
                    val_batches += 1
            
            val_loss /= len(val_loader.dataset)
            val_acc_global = val_correct / val_total
            val_iou /= val_batches
            val_acc /= val_batches
            
            # Prepare and print validation metrics table
            val_table = [[class_names[i], f"{val_iou[i]:.2f}", f"{val_acc[i]:.2f}"] for i in range(num_classes)]
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc_global:.4f}")
            print("Validation Metrics:")
            print(tabulate(val_table, headers=["Class", "IoU", "Acc"], tablefmt="grid"))
            
            if val_loss < best_val_loss and val_batches > 0:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_hooknet.pth")
                logging.info(f"Epoch {epoch+1}/{num_epochs}, Saved best model with Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, No validation data provided")

    return model