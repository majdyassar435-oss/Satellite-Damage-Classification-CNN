import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------
# 1. إعداد المتغيرات الأساسية
# ---------------------------------------------------------
IMG_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 35

TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/val'

def strict_clean_dataset(directory):
    """تنظيف الصور وإعادة حفظها بتنسيق PNG مع الحفاظ على ألوان RGB الصحيحة"""
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # قراءة الصورة
            img = cv2.imread(file_path)
            if img is None:
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            else:
                # التحويل من BGR إلى RGB قبل الحفظ لضمان سلامة الألوان
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                new_file_path = os.path.splitext(file_path)[0] + ".png"
                
                # حفظ الصورة بنظام RGB صحيح عبر OpenCV (نحولها لـ BGR فقط عند الكتابة على القرص)
                cv2.imwrite(new_file_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                
                if file_path != new_file_path:
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass

def train():
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print("❌ خطأ: مجلدات البيانات غير موجودة.")
        return

    print("🧹 جاري التنظيف الصارم وإعادة تنميط الصور مع ضبط الألوان...")
    strict_clean_dataset(TRAIN_DIR)
    strict_clean_dataset(VAL_DIR)

    print("🔄 جاري تحميل البيانات من المجلدات...")
    
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int'
    )

    val_ds = keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int'
    )

    print("📁 الفئات المكتشفة بالترتيب:")
    print(train_ds.class_names)
    print("⚠️ تأكد أن هذا الترتيب يطابق تماماً مصفوفة CLASS_NAMES في app.py!")

    # =========================================================
    # 2. بناء معمارية النموذج مع Data Augmentation و Rescaling
    # =========================================================
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
    ])

    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        data_augmentation, 
        layers.Rescaling(1./255),  # معايرة القيم داخل النموذج
        
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # Output Layer (5 الفئات)
        layers.Dense(5, activation='softmax')
    ])
    # =========================================================

    # 3. تجميع النموذج
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. بدء التدريب
    print("🚀 بدء إعادة تدريب النموذج بالتحسينات الجديدة...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # 5. حفظ النموذج
    model.save('war_debris_5class_model.h5')
    print("✅ تم إعادة تدريب النموذج وحفظه بنجاح!")

if __name__ == '__main__':
    train()