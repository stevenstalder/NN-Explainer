import torch
import torchmetrics

class SingleLabelMetrics(torchmetrics.Metric):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.add_state("true_positives", torch.zeros(num_classes))
        self.add_state("false_positives", torch.zeros(num_classes))
        self.add_state("true_negatives", torch.zeros(num_classes))
        self.add_state("false_negatives", torch.zeros(num_classes))

    def update(self, logits, labels):
        with torch.no_grad():
            for i, batch_sample_logits in enumerate(logits):
                self.true_negatives += 1.0
                top_class_prediction = batch_sample_logits.argmax(-1)
                if labels[i] == top_class_prediction:
                    self.true_positives[labels] += 1.0
                    self.true_negatives[labels] -= 1.0
                else:
                    self.false_negatives[labels] += 1.0
                    self.false_positives[top_class_prediction] += 1.0
                    self.true_negatives[labels] -= 1.0
                    self.true_negatives[top_class_prediction] -= 1.0

    def compute(self):
        accuracy = ((self.true_positives + self.true_negatives) / (self.true_positives + self.true_negatives + self.false_positives + self.false_negatives)).mean()
        precision = (self.true_positives / (self.true_positives + self.false_positives)).mean()
        recall = (self.true_positives / (self.true_positives + self.false_negatives)).mean()
        f_score = ((2 * self.true_positives) / (2 * self.true_positives + self.false_positives + self.false_negatives)).mean()

        return {'Accuracy': accuracy.item(), 'Precision': precision.item(), 'Recall': recall.item(), 'F-Score': f_score.item()}

class MultiLabelMetrics(torchmetrics.Metric):
    def __init__(self, num_classes, threshold):
        super().__init__()

        self.num_classes = num_classes
        self.threshold = threshold

        self.add_state("true_positives", torch.tensor(0.0))
        self.add_state("false_positives", torch.tensor(0.0))
        self.add_state("true_negatives", torch.tensor(0.0))
        self.add_state("false_negatives", torch.tensor(0.0))

    def update(self, logits, labels):
        with torch.no_grad():
            for i, batch_sample_logits in enumerate(logits):
                for j in range(self.num_classes):
                    if labels[i][j] == 1.0:
                        if batch_sample_logits[j] >= self.threshold:
                            self.true_positives += 1.0
                        else:
                            self.false_negatives += 1.0
                    else:
                        if batch_sample_logits[j] >= self.threshold:
                            self.false_positives += 1.0
                        else:
                            self.true_negatives += 1.0

    def compute(self):
        self.accuracy = ((self.true_positives + self.true_negatives) / (self.true_positives + self.true_negatives + self.false_positives + self.false_negatives))
        self.precision = (self.true_positives / (self.true_positives + self.false_positives))
        self.recall = (self.true_positives / (self.true_positives + self.false_negatives))
        self.f_score = ((2 * self.true_positives) / (2 * self.true_positives + self.false_positives + self.false_negatives))

        return {'Accuracy': self.accuracy.item(), 'Precision': self.precision.item(), 'Recall': self.recall.item(), 'F-Score': self.f_score.item()}

    def save(self, model, classifier_type, dataset):
        f = open(model + "_" + classifier_type + "_" + dataset + "_" + "test_metrics.txt", "w")
        f.write("Accuracy: " + str(self.accuracy.item()) + "\n")
        f.write("Precision: " + str(self.precision.item()) + "\n")
        f.write("Recall: " + str(self.recall.item()) + "\n")
        f.write("F-Score: " + str(self.f_score.item()))
        f.close()

### BELOW ARE JUST UTILITY FUNCTIONS, NOT THE ONES USED FOR THE RESULTS IN THE PAPER/THESIS ###

class MultiLabelTopPredictionAccuracy(torchmetrics.Metric):
    def __init__(self):
        super().__init__()

        self.add_state("correct", torch.tensor(0.0))
        self.add_state("total", torch.tensor(0.0))

    def update(self, logits, targets):
        with torch.no_grad():
            for i, batch_sample_logits in enumerate(logits):
                self.total += 1.0
                top_class_prediction = batch_sample_logits.argmax(-1)
                if (targets[i][top_class_prediction] == 1.0):
                    self.correct += 1.0

    def compute(self):
        return {'Top prediction accuracy': self.correct / self.total}

class MultiLabelPrecision(torchmetrics.Metric):
    def __init__(self, num_classes, threshold):
        super().__init__()

        self.num_classes = num_classes
        self.threshold = threshold

        self.add_state("true_positives", torch.tensor(0.0))
        self.add_state("false_positives", torch.tensor(0.0))

    def update(self, logits, targets):
        with torch.no_grad():
            for i, batch_sample_logits in enumerate(logits):
                for j in range(self.num_classes):
                    if (batch_sample_logits[j] >= self.threshold):
                        if (targets[i][j] == 1.0):
                            self.true_positives += 1.0
                        else:
                            self.false_positives += 1.0

    def compute(self):
        return self.true_positives / (self.true_positives + self.false_positives)

class MultiLabelRecall(torchmetrics.Metric):
    def __init__(self, num_classes, threshold):
        super().__init__()

        self.num_classes = num_classes
        self.threshold = threshold

        self.add_state("true_positives", torch.tensor(0.0))
        self.add_state("false_negatives", torch.tensor(0.0))

    def update(self, logits, targets):
        with torch.no_grad():
            for i, batch_sample_logits in enumerate(logits):
                for j in range(self.num_classes):
                    if (targets[i][j] == 1.0):
                        if (batch_sample_logits[j] >= self.threshold):
                            self.true_positives += 1.0
                        else:
                            self.false_negatives += 1.0

