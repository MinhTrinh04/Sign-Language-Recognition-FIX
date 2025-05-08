import numpy as np
import pickle
from collections import Counter # Để tìm nhãn phổ biến nhất

# --- Hàm train_test_split tự custom ---
def custom_train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    """
    Hàm chia dữ liệu X, y thành tập huấn luyện và tập kiểm tra.
    Lưu ý: Phiên bản này chưa hỗ trợ 'stratify'.
    """
    if random_state:
        np.random.seed(random_state)
    
    dataset_size = len(X)
    indices = np.arange(dataset_size)
    
    if shuffle:
        np.random.shuffle(indices)
        
    split_idx = int(dataset_size * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

# --- Hàm accuracy_score tự custom ---
def custom_accuracy_score(y_true, y_pred):
    """Tính toán độ chính xác."""
    return np.sum(y_true == y_pred) / len(y_true)

# --- Hàm Classification Report tự custom (Phiên bản đơn giản) ---
def custom_classification_report_simplified(y_true, y_pred):
    """Tạo báo cáo phân loại đơn giản (Precision, Recall, F1 cho mỗi lớp)."""
    report_lines = ["Custom Classification Report (Simplified):"]
    unique_labels = sorted(np.unique(np.concatenate((y_true, y_pred)))) # Sắp xếp để báo cáo nhất quán
    
    for label in unique_labels:
        tp = np.sum((y_true == label) & (y_pred == label)) # True Positives
        fp = np.sum((y_true != label) & (y_pred == label)) # False Positives
        fn = np.sum((y_true == label) & (y_pred != label)) # False Negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = np.sum(y_true == label)
        
        report_lines.append(f"\nLabel '{label}':")
        report_lines.append(f"  Precision: {precision:.4f}")
        report_lines.append(f"  Recall:    {recall:.4f}")
        report_lines.append(f"  F1-score:  {f1_score:.4f}")
        report_lines.append(f"  Support:   {support}")
        
    overall_accuracy = custom_accuracy_score(y_true, y_pred)
    report_lines.append(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    return "\n".join(report_lines)

# --- Lớp Node cho Cây Quyết Định ---
class Node:
    def __init__(self, feature_index=None, threshold=None, left_child=None, right_child=None, *, value=None):
        """
        feature_index: Chỉ số của thuộc tính dùng để chia.
        threshold: Ngưỡng giá trị để chia.
        left_child: Node con trái.
        right_child: Node con phải.
        value: Giá trị (nhãn) nếu đây là node lá.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.value = value # Giá trị của node lá

    def is_leaf_node(self):
        return self.value is not None

# --- Lớp DecisionTreeClassifier tự custom ---
class CustomDecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=100, n_features_to_consider=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        # Số lượng feature sẽ được xem xét tại mỗi split (quan trọng cho Random Forest)
        self.n_features_to_consider = n_features_to_consider
        self.root_node = None

    def _calculate_gini_impurity(self, y):
        """Tính Gini impurity cho một tập nhãn y."""
        if len(y) == 0:
            return 0
        # Counter(y).values() trả về số lần xuất hiện của mỗi nhãn
        class_counts = np.array(list(Counter(y).values()))
        probabilities = class_counts / len(y)
        gini = 1 - np.sum(probabilities**2)
        return gini

    def _find_best_split(self, X, y, feature_indices):
        """Tìm cách chia (thuộc tính, ngưỡng) tốt nhất dựa trên Gini impurity."""
        best_gini_reduction = -1 # Ta muốn Gini sau khi chia là nhỏ nhất, tức là Gini reduction lớn nhất
                                # Hoặc tìm Gini có trọng số nhỏ nhất. Giả sử tìm Gini có trọng số nhỏ nhất.
        best_weighted_gini = 1.0 # Gini luôn <= 1

        split_info = {'feature_index': None, 'threshold': None, 
                      'left_indices': None, 'right_indices': None}

        current_gini = self._calculate_gini_impurity(y)

        for feature_idx in feature_indices:
            unique_thresholds = np.unique(X[:, feature_idx])
            for threshold in unique_thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = X[:, feature_idx] > threshold

                if not np.any(left_mask) or not np.any(right_mask):
                    continue # Bỏ qua nếu việc chia tạo ra một nhánh rỗng

                y_left, y_right = y[left_mask], y[right_mask]
                
                # Tính Gini impurity có trọng số
                p_left = len(y_left) / len(y)
                p_right = len(y_right) / len(y)
                weighted_gini = p_left * self._calculate_gini_impurity(y_left) + \
                                p_right * self._calculate_gini_impurity(y_right)

                if weighted_gini < best_weighted_gini:
                    best_weighted_gini = weighted_gini
                    split_info['feature_index'] = feature_idx
                    split_info['threshold'] = threshold
                    split_info['left_indices'] = np.where(left_mask)[0]
                    split_info['right_indices'] = np.where(right_mask)[0]
        
        # Nếu không có split nào cải thiện (best_weighted_gini vẫn là 1.0 hoặc không giảm nhiều so với current_gini)
        # thì không chia nữa (trả về None cho feature_index).
        # Hoặc, một tiêu chí khác: nếu best_weighted_gini >= current_gini (không có sự cải thiện)
        if best_weighted_gini >= current_gini and best_weighted_gini != 0 : # Thêm điều kiện !=0 để tránh trường hợp node đã pure
             split_info['feature_index'] = None


        return split_info

    def _build_decision_tree(self, X, y, current_depth=0):
        """Xây dựng cây quyết định một cách đệ quy."""
        n_samples, n_total_features = X.shape
        
        # Điều kiện dừng đệ quy
        if (current_depth >= self.max_depth or
            len(np.unique(y)) == 1 or  # Node đã thuần khiết
            n_samples < self.min_samples_split):
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # Chọn ngẫu nhiên một tập con các thuộc tính để xem xét (cho Random Forest)
        if self.n_features_to_consider is None:
            # Nếu không chỉ định, xem xét tất cả các thuộc tính (hành vi của Decision Tree thông thường)
            feature_indices_to_try = np.arange(n_total_features)
        else:
            feature_indices_to_try = np.random.choice(n_total_features, self.n_features_to_consider, replace=False)
        
        split_info = self._find_best_split(X, y, feature_indices_to_try)

        # Nếu không tìm được cách chia tốt (ví dụ, không giảm Gini)
        if split_info['feature_index'] is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # Đệ quy xây dựng cây con trái và phải
        left_child_node = self._build_decision_tree(X[split_info['left_indices']], 
                                                    y[split_info['left_indices']], 
                                                    current_depth + 1)
        right_child_node = self._build_decision_tree(X[split_info['right_indices']], 
                                                     y[split_info['right_indices']], 
                                                     current_depth + 1)
        
        return Node(feature_index=split_info['feature_index'], 
                    threshold=split_info['threshold'], 
                    left_child=left_child_node, 
                    right_child=right_child_node)

    def fit(self, X, y):
        """Huấn luyện cây quyết định."""
        # Nếu n_features_to_consider không được set cho RF, DT sẽ dùng tất cả features
        if self.n_features_to_consider is None or self.n_features_to_consider > X.shape[1]:
            self.n_features_to_consider = X.shape[1]
        self.root_node = self._build_decision_tree(X, y)

    def _traverse_decision_tree(self, x_sample, current_node):
        """Duyệt cây để dự đoán cho một mẫu x_sample."""
        if current_node.is_leaf_node():
            return current_node.value
        
        if x_sample[current_node.feature_index] <= current_node.threshold:
            return self._traverse_decision_tree(x_sample, current_node.left_child)
        else:
            return self._traverse_decision_tree(x_sample, current_node.right_child)

    def predict(self, X):
        """Dự đoán nhãn cho tập dữ liệu X."""
        predictions = [self._traverse_decision_tree(x_sample, self.root_node) for x_sample in X]
        return np.array(predictions)

# --- Lớp RandomForestClassifier tự custom ---
class CustomRandomForestClassifier:
    def __init__(self, n_estimators=100, min_samples_split=2, max_depth=100, n_features_subsample=None):
        self.n_estimators = n_estimators # Số lượng cây
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        # Số lượng feature con để mỗi cây Decision Tree xem xét. Thường là sqrt(tổng số features).
        self.n_features_subsample = n_features_subsample 
        self.decision_trees = []
        self.classes_ = None # Để lưu các lớp duy nhất khi fit, giống sklearn

    def _create_bootstrap_sample(self, X, y):
        """Tạo một mẫu bootstrap từ X, y."""
        n_samples = X.shape[0]
        # Lấy ngẫu nhiên các chỉ số (có hoàn lại)
        random_indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[random_indices], y[random_indices]

    def fit(self, X, y):
        """Huấn luyện mô hình Random Forest."""
        self.decision_trees = []
        self.classes_ = np.unique(y) # Lưu lại các lớp đã thấy
        
        # Xác định số lượng feature con nếu chưa được cung cấp
        if self.n_features_subsample is None:
            self.n_features_subsample = int(np.sqrt(X.shape[1]))
        # Đảm bảo không lớn hơn tổng số feature
        self.n_features_subsample = min(self.n_features_subsample, X.shape[1])


        for _ in range(self.n_estimators):
            decision_tree = CustomDecisionTreeClassifier(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features_to_consider=self.n_features_subsample # Đây là phần "random" của feature
            )
            X_bootstrap, y_bootstrap = self._create_bootstrap_sample(X, y)
            decision_tree.fit(X_bootstrap, y_bootstrap)
            self.decision_trees.append(decision_tree)

    def predict(self, X):
        """Dự đoán nhãn cho X bằng cách tổng hợp kết quả từ các cây (majority vote)."""
        # Lấy dự đoán từ tất cả các cây
        # tree_predictions có dạng (n_estimators, n_samples)
        tree_predictions = np.array([tree.predict(X) for tree in self.decision_trees])
        
        # Chuyển vị để có dạng (n_samples, n_estimators)
        tree_predictions_transposed = tree_predictions.T
        
        # Thực hiện majority vote cho từng mẫu
        final_predictions = [Counter(preds_for_sample).most_common(1)[0][0] for preds_for_sample in tree_predictions_transposed]
        return np.array(final_predictions)

# --- Logic chính của train_classifier.py ---
if __name__ == '__main__':
    print("Đang tải dữ liệu từ data.pickle...")
    data_dict = pickle.load(open('./data.pickle', 'rb'))
    
    data_list = data_dict['data'] # data is a list of lists/arrays
    labels_list = data_dict['labels']

    # Đảm bảo tất cả các mẫu dữ liệu có cùng độ dài (ví dụ: 42 features)
    # và chuyển đổi thành NumPy array phù hợp.
    # Việc kiểm tra và chuẩn hóa dữ liệu nên được thực hiện kỹ ở create_dataset.py
    # Ở đây, chúng ta giả định rằng các mẫu đã gần đúng, và sẽ lọc/pad nếu cần.
    
    processed_data = []
    processed_labels = []
    expected_feature_length = 42 # Độ dài đặc trưng mong muốn

    for i, sample in enumerate(data_list):
        sample_arr = np.array(sample)
        if len(sample_arr) == expected_feature_length:
            processed_data.append(sample_arr)
            processed_labels.append(labels_list[i])
        elif len(sample_arr) > expected_feature_length:
            # Cân nhắc: cắt bớt hay bỏ qua? Hiện tại là cắt bớt.
            # print(f"Cảnh báo: Mẫu {i+1} (nhãn: {labels_list[i]}) có độ dài {len(sample_arr)}, được cắt thành {expected_feature_length}.")
            processed_data.append(sample_arr[:expected_feature_length])
            processed_labels.append(labels_list[i])
        else: # len(sample_arr) < expected_feature_length
            # Cân nhắc: pad bằng 0 hay bỏ qua? Hiện tại là bỏ qua.
            print(f"Cảnh báo: Mẫu {i+1} (nhãn: {labels_list[i]}) có độ dài {len(sample_arr)}, nhỏ hơn {expected_feature_length}. Mẫu này bị bỏ qua.")
            pass # Bỏ qua mẫu không đủ độ dài

    if not processed_data:
        print("Lỗi: Không có dữ liệu hợp lệ để huấn luyện sau khi xử lý. Kết thúc.")
        exit()

    data = np.asarray(processed_data)
    labels = np.asarray(processed_labels)

    print(f"Tổng số mẫu hợp lệ sau xử lý: {len(data)}")

    # Chia dữ liệu bằng hàm tự custom
    x_train, x_test, y_train, y_test = custom_train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=42)
    print(f"Kích thước tập huấn luyện: X_train {x_train.shape}, y_train {y_train.shape}")
    print(f"Kích thước tập kiểm tra: X_test {x_test.shape}, y_test {y_test.shape}")

    # Sử dụng RandomForestClassifier tự custom
    # Bạn có thể điều chỉnh các tham số này
    model = CustomRandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2) 
    
    print("\nBắt đầu huấn luyện mô hình Custom Random Forest...")
    model.fit(x_train, y_train)
    print("Huấn luyện mô hình hoàn tất.")

    # Dự đoán trên tập kiểm tra
    y_predict = model.predict(x_test)

    # Tính toán độ chính xác bằng hàm tự custom
    # Lưu ý: thứ tự đúng là (y_true, y_pred)
    score = custom_accuracy_score(y_test, y_predict) 

    print(f"\n{score*100:.2f}% mẫu trong tập kiểm tra được phân loại chính xác!")

    # Lưu mô hình tự custom
    # Quan trọng: Khi tải lại model này, các định nghĩa lớp CustomRandomForestClassifier, 
    # CustomDecisionTreeClassifier, và Node phải có sẵn.
    model_filename = 'custom_random_forest_model.p'
    with open(model_filename, 'wb') as f:
        pickle.dump({'model': model}, f)
    print(f"Mô hình tự custom đã được lưu vào '{model_filename}'")

    # In báo cáo phân loại (phiên bản đơn giản)
    print("\nBáo cáo phân loại (phiên bản đơn giản hóa):")
    print(custom_classification_report_simplified(y_test, y_predict))