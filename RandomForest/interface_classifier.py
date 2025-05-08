import cv2
import mediapipe as mp
import pickle
import numpy as np

# Bước 1: Import các lớp và hàm cần thiết từ module tự tạo
# Đảm bảo tệp 'custom_ml_algorithms.py' nằm cùng thư mục hoặc trong PYTHONPATH
from custom_ml_algorithms import CustomRandomForestClassifier, CustomDecisionTreeClassifier, Node
# Các hàm custom_accuracy_score, custom_train_test_split, custom_classification_report_simplified
# không trực tiếp cần thiết ở đây cho việc dự đoán, nhưng các lớp Node, CustomDecisionTreeClassifier,
# CustomRandomForestClassifier là BẮT BUỘC để pickle có thể tải model.

# Bước 2: Tải model tự custom
# Thay 'custom_random_forest_model.p' bằng tên tệp model bạn đã lưu từ train_classifier.py (phiên bản custom)
try:
    with open('custom_random_forest_model.p', 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
    print("Model tự custom đã được tải thành công.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy tệp model 'custom_random_forest_model.p'.")
    print("Hãy đảm bảo bạn đã huấn luyện và lưu model bằng train_classifier.py (phiên bản custom).")
    exit()
except Exception as e:
    print(f"Lỗi khi tải model: {e}")
    print("Hãy đảm bảo các định nghĩa lớp (Node, CustomDecisionTreeClassifier, CustomRandomForestClassifier) có sẵn và chính xác.")
    exit()


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Lỗi: Không thể mở webcam.")
    exit()

mp_hand = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hand.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Bước 3: label_dict
# Kiểm tra xem model.classes_ có tồn tại và chứa các nhãn mong muốn không
# (Lớp CustomRandomForestClassifier của chúng ta có lưu self.classes_ từ dữ liệu huấn luyện)
# Bạn cần đảm bảo rằng label_dict này ánh xạ ĐÚNG các ký tự mà model của bạn dự đoán.
# Ví dụ, nếu model dự đoán ra các số 0, 1, 2,... thì label_dict[0] phải là 'A', v.v.
# Hoặc nếu model dự đoán trực tiếp ra 'A', 'B', 'C',... thì label_dict[prediction[0]] là được.

label_dict = {}
if hasattr(model, 'classes_') and model.classes_ is not None:
    print(f"Các lớp mà model đã học: {model.classes_}")
    # Giả sử model.classes_ là ['A', 'B', 'C', ...] hoặc ['0', '1', '2', ...]
    # và bạn muốn ánh xạ chúng tới chính nó hoặc một giá trị khác.
    # Dưới đây là ví dụ nếu model dự đoán trực tiếp ra ký tự (A-Z)
    for i in range(26): # A-Z
        char_label = chr(65 + i)
        label_dict[char_label] = char_label
    # label_dict['V'] = 'Hello everyone <33' # Nếu bạn có nhãn đặc biệt này trong model
    # Hoặc nếu model dự đoán số, bạn cần ánh xạ lại cho phù hợp
    # Ví dụ: for i, cls_name in enumerate(model.classes_): label_dict[cls_name] = chr(65 + i) # Nếu cls_name là số 0, 1,..
else:
    print("Cảnh báo: Model không có thuộc tính 'classes_'. Sử dụng label_dict mặc định.")
    # Tạo label_dict mặc định (A-Z) - Cẩn thận nếu model của bạn không dự đoán theo kiểu này
    for i in range(26):
        label_dict[chr(65 + i)] = chr(65 + i)
    # label_dict['V'] = 'Hello everyone <33'

print(f"Sử dụng label_dict: {label_dict}")


while True:
    data_aux = []
    x_coords = [] # Đổi tên x_ thành x_coords để rõ ràng hơn
    y_coords = [] # Đổi tên y_ thành y_coords

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Lỗi đọc frame hoặc kết thúc video.")
        break

    H, W, _ = frame.shape
    # frame_flipped = cv2.flip(frame, 0) # Biến này không được dùng

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Chỉ xử lý bàn tay đầu tiên để đảm bảo 42 đặc trưng
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hand.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        for i in range(len(hand_landmarks.landmark)): # Nên là 21 điểm mốc
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.extend([x, y]) # Thêm cả x và y vào data_aux
            x_coords.append(x)
            y_coords.append(y)

        # Đảm bảo data_aux có đúng 42 đặc trưng (21 cặp x,y)
        # Việc này rất quan trọng cho đầu vào của model
        expected_feature_length = 42
        if len(data_aux) > expected_feature_length:
            data_aux = data_aux[:expected_feature_length]
            # x_coords và y_coords cũng nên được cắt nếu cần, dù chúng chủ yếu dùng để vẽ bounding box
            x_coords = x_coords[:expected_feature_length//2]
            y_coords = y_coords[:expected_feature_length//2]
        elif len(data_aux) < expected_feature_length and len(data_aux) > 0 : # Nếu thiếu, pad bằng 0
             padding = [0.0] * (expected_feature_length - len(data_aux))
             data_aux.extend(padding)


        if len(data_aux) == expected_feature_length and x_coords and y_coords: # Chỉ dự đoán nếu có đủ dữ liệu
            # Tính toán bounding box
            x1 = int(min(x_coords) * W) - 10 # Giảm lề một chút nếu muốn box sát hơn
            y1 = int(min(y_coords) * H) - 10
            x2 = int(max(x_coords) * W) + 10
            y2 = int(max(y_coords) * H) + 10

            # Dự đoán bằng model tự custom
            # Model của chúng ta nhận đầu vào là mảng NumPy 2D (dù chỉ 1 mẫu)
            prediction_input = np.asarray(data_aux).reshape(1, -1)
            predicted_value_array = model.predict(prediction_input) # model.predict trả về mảng NumPy
            
            if predicted_value_array.size > 0:
                predicted_value = predicted_value_array[0] # Lấy phần tử đầu tiên
                
                # Tra cứu trong label_dict
                # Sử dụng .get() để tránh KeyError nếu predicted_value không có trong dict
                predicted_character = label_dict.get(str(predicted_value), "?") # Chuyển predicted_value sang str nếu nó là số

                # print(f"Input: {prediction_input}, Predicted value: {predicted_value}, Character: {predicted_character}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)
            else:
                print("Model không đưa ra dự đoán.")
        # else:
            # print(f"Không đủ dữ liệu để dự đoán. Len data_aux: {len(data_aux)}")


    cv2.imshow('frame', frame)
    key = cv2.waitKey(20) # Tăng thời gian chờ một chút
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()