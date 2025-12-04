package Model;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;

public class RandomForestClassifier {

    private static class TreeNode {
        boolean isLeaf;
        String label;
        int featureIndex;
        double threshold;
        TreeNode left;
        TreeNode right;
    }

    private List<TreeNode> trees;
    private int numTrees;
    private int maxDepth;
    private int minSamplesSplit;
    private int maxFeatures;
    private Random random;

    public RandomForestClassifier(int numTrees, int maxDepth,
            int minSamplesSplit, int maxFeatures) {
        this.numTrees = numTrees;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.maxFeatures = maxFeatures;
        this.trees = new ArrayList<>();
        this.random = new Random(42);
    }

    // ================== API TRAIN/PREDICT/ACCURACY ==================

    public void fit(List<double[]> X, List<String> y) {
        trees.clear();
        int nSamples = X.size();
        if (nSamples == 0)
            return;

        int nFeatures = X.get(0).length;
        int mFeatures = maxFeatures > 0 ? Math.min(maxFeatures, nFeatures) : nFeatures;

        for (int t = 0; t < numTrees; t++) {
            // Bootstrap sample
            List<double[]> sampleX = new ArrayList<>();
            List<String> sampleY = new ArrayList<>();
            for (int i = 0; i < nSamples; i++) {
                int idx = random.nextInt(nSamples);
                sampleX.add(X.get(idx));
                sampleY.add(y.get(idx));
            }
            TreeNode root = buildTree(sampleX, sampleY, 0, mFeatures);
            trees.add(root);
        }
    }

    public String predict(double[] x) {
        if (trees.isEmpty())
            return null;

        Map<String, Integer> votes = new HashMap<>();
        for (TreeNode tree : trees) {
            String label = predictTree(tree, x);
            if (label == null)
                continue;
            votes.put(label, votes.getOrDefault(label, 0) + 1);
        }

        String bestLabel = null;
        int bestCount = -1;
        for (Map.Entry<String, Integer> e : votes.entrySet()) {
            if (e.getValue() > bestCount) {
                bestCount = e.getValue();
                bestLabel = e.getKey();
            }
        }
        return bestLabel;
    }

    public double accuracy(List<double[]> Xtest, List<String> ytest) {
        if (Xtest.isEmpty())
            return 0.0;
        int correct = 0;
        for (int i = 0; i < Xtest.size(); i++) {
            String pred = predict(Xtest.get(i));
            if (pred != null && pred.equals(ytest.get(i))) {
                correct++;
            }
        }
        return (double) correct / (double) Xtest.size();
    }

    public static void runRandomForest(String trainCsvPath) {
        try {
            List<double[]> X = new ArrayList<>();
            List<String> y = new ArrayList<>();
            loadData(trainCsvPath, X, y);

            if (X.isEmpty()) {
                System.out.println("No data loaded from " + trainCsvPath);
                return;
            }

            int n = X.size();
            List<String> classLabels = getUniqueLabels(y);

            // Shuffle + chia 80/20 nội bộ
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < n; i++)
                indices.add(i);
            Collections.shuffle(indices, new Random(42));

            int splitIndex = (int) (0.8 * n);
            List<double[]> X_train = new ArrayList<>();
            List<String> y_train = new ArrayList<>();
            List<double[]> X_test = new ArrayList<>();
            List<String> y_test = new ArrayList<>();

            for (int i = 0; i < n; i++) {
                int idx = indices.get(i);
                if (i < splitIndex) {
                    X_train.add(X.get(idx));
                    y_train.add(y.get(idx));
                } else {
                    X_test.add(X.get(idx));
                    y_test.add(y.get(idx));
                }
            }

            int nFeatures = X_train.get(0).length;
            int numTrees = 50;
            int maxDepth = 10;
            int minSamplesSplit = 5;
            int maxFeatures = (int) Math.sqrt(nFeatures);

            RandomForestClassifier rf = new RandomForestClassifier(
                    numTrees, maxDepth, minSamplesSplit, maxFeatures);

            // Đo Training Time
            long startTime = System.currentTimeMillis();
            rf.fit(X_train, y_train);
            long trainingTime = System.currentTimeMillis() - startTime;

            // Đo Prediction Time
            startTime = System.currentTimeMillis();
            List<String> y_pred = new ArrayList<>();
            for (double[] x : X_test) {
                y_pred.add(rf.predict(x));
            }
            long predictionTime = System.currentTimeMillis() - startTime;

            double acc = rf.accuracy(X_test, y_test);

            // Tính toán Metrics
            Map<String, Map<String, Double>> metrics = calculateMetrics(y_test, y_pred, classLabels);
            Map<String, Double> cmData = metrics.get("Confusion Matrix");

            System.out.println("===== RANDOM FOREST CLASSIFICATION =====");
            System.out.println("Train csv path: " + trainCsvPath);
            System.out.println("Total samples: " + n);
            System.out.println("Train samples: " + X_train.size());
            System.out.println("Test samples:  " + X_test.size());
            System.out.println("Num trees:     " + numTrees);
            System.out.println("Max depth:     " + maxDepth);
            System.out.println("Max features:  " + maxFeatures);
            System.out.println("Runtime (Training): " + trainingTime + " ms");
            System.out.println("Runtime (Prediction): " + predictionTime + " ms");
            System.out.println("Accuracy (Overall) = " + String.format("%.4f", acc));

            System.out.println("\n========== CLASSIFICATION REPORT ==========");
            System.out.printf("| %-8s | %-10s | %-7s | %-9s | %-7s |\n",
                    "Class", "Precision", "Recall", "F1-Score", "Support");
            System.out.println("|:--------:|:----------:|:-------:|:---------:|:-------:|");

            for (String label : classLabels) {
                Map<String, Double> m = metrics.get(label);
                System.out.printf("| %-8s | %-10s | %-7s | %-9s | %-7s |\n",
                        label,
                        String.format("%.4f", m.get("Precision")),
                        String.format("%.4f", m.get("Recall")),
                        String.format("%.4f", m.get("F1-Score")),
                        String.format("%.0f", m.get("Support")));
            }

            System.out.println("|------------------------------------------------------------|");
            Map<String, Double> macro = metrics.get("Macro Avg");
            System.out.printf("| %-8s | %-10s | %-7s | %-9s | %-7s |\n",
                    "Macro Avg",
                    String.format("%.4f", macro.get("Precision")),
                    String.format("%.4f", macro.get("Recall")),
                    String.format("%.4f", macro.get("F1-Score")),
                    String.format("%.0f", macro.get("Support")));

            System.out.println("\n========== CONFUSION MATRIX (Predicted vs Actual) ==========");
            System.out.printf("| %-8s", "|");
            for (String label : classLabels) {
                System.out.printf(" %-8s |", label + " (P)");
            }
            System.out.println();

            System.out.printf("|:--------:", "|");
            for (int i = 0; i < classLabels.size(); i++) {
                System.out.printf(":--------:|");
            }
            System.out.println();

            for (String actualLabel : classLabels) {
                System.out.printf("| %-8s |", actualLabel + " (A)");
                for (String predictedLabel : classLabels) {
                    String key = actualLabel + "->" + predictedLabel;
                    System.out.printf(" %-8.0f |", cmData.getOrDefault(key, 0.0));
                }
                System.out.println();
            }

            System.out.println("===== END RANDOM FOREST =====");

        } catch (IOException e) {
            System.err.println("Error reading training data: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void loadData(String path,
            List<double[]> X,
            List<String> y) throws IOException {

        List<String> lines = Files.readAllLines(Paths.get(path));
        if (lines.isEmpty())
            return;

        // Bỏ qua dòng headers
        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty())
                continue;

            String[] parts = line.split(",", -1);
            if (parts.length < 3)
                continue;

            // Lấy nhãn TRỰC TIẾP từ cột 2 (Life Ladder)
            // Nhãn đã là "Low", "Medium", "High"
            String label = parts[2].trim();
            y.add(label);

            double[] features = new double[parts.length - 2];
            int idx = 0;

            features[idx++] = Double.parseDouble(parts[1]);

            for (int j = 3; j < parts.length; j++) {
                if (parts[j].isEmpty()) {

                    features[idx++] = 0.0;
                } else {

                    features[idx++] = Double.parseDouble(parts[j]);
                }
            }
            X.add(features);
        }
    }

    private TreeNode buildTree(List<double[]> X, List<String> y,
            int depth, int mFeatures) {
        TreeNode node = new TreeNode();
        String majority = majorityLabel(y);

        // điều kiện dừng
        if (depth >= maxDepth || X.size() < minSamplesSplit || isPure(y)) {
            node.isLeaf = true;
            node.label = majority;
            return node;
        }

        int nFeatures = X.get(0).length;
        int[] featureIndices = getRandomFeatureIndices(nFeatures, mFeatures);

        double bestGini = Double.MAX_VALUE;
        int bestFeature = -1;
        double bestThreshold = Double.NaN;
        List<double[]> bestLeftX = null;
        List<String> bestLeftY = null;
        List<double[]> bestRightX = null;
        List<String> bestRightY = null;

        // duyệt các feature được chọn
        for (int fIndex : featureIndices) {
            double[] values = new double[X.size()];
            for (int i = 0; i < X.size(); i++) {
                values[i] = X.get(i)[fIndex];
            }
            double[] sorted = values.clone();
            Arrays.sort(sorted);

            List<Double> thresholds = new ArrayList<>();
            for (int i = 1; i < sorted.length; i++) {
                if (sorted[i] != sorted[i - 1]) {
                    thresholds.add((sorted[i] + sorted[i - 1]) / 2.0);
                }
            }
            if (thresholds.isEmpty())
                continue;

            for (Double thrObj : thresholds) {
                double thr = thrObj;
                List<double[]> leftX = new ArrayList<>();
                List<String> leftY = new ArrayList<>();
                List<double[]> rightX = new ArrayList<>();
                List<String> rightY = new ArrayList<>();

                for (int i = 0; i < X.size(); i++) {
                    if (X.get(i)[fIndex] <= thr) {
                        leftX.add(X.get(i));
                        leftY.add(y.get(i));
                    } else {
                        rightX.add(X.get(i));
                        rightY.add(y.get(i));
                    }
                }
                if (leftX.isEmpty() || rightX.isEmpty())
                    continue;

                double gini = giniImpurity(leftY, rightY);
                if (gini < bestGini) {
                    bestGini = gini;
                    bestFeature = fIndex;
                    bestThreshold = thr;
                    bestLeftX = leftX;
                    bestLeftY = leftY;
                    bestRightX = rightX;
                    bestRightY = rightY;
                }
            }
        }

        if (bestFeature == -1) {
            node.isLeaf = true;
            node.label = majority;
            return node;
        }

        node.isLeaf = false;
        node.featureIndex = bestFeature;
        node.threshold = bestThreshold;
        node.label = majority;
        node.left = buildTree(bestLeftX, bestLeftY, depth + 1, mFeatures);
        node.right = buildTree(bestRightX, bestRightY, depth + 1, mFeatures);
        return node;
    }

    private String majorityLabel(List<String> y) {
        Map<String, Integer> counts = new HashMap<>();
        for (String label : y) {
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }
        String bestLabel = null;
        int bestCount = -1;
        for (Map.Entry<String, Integer> e : counts.entrySet()) {
            if (e.getValue() > bestCount) {
                bestCount = e.getValue();
                bestLabel = e.getKey();
            }
        }
        return bestLabel;
    }

    private boolean isPure(List<String> y) {
        if (y.isEmpty())
            return true;
        String first = y.get(0);
        for (int i = 1; i < y.size(); i++) {
            if (!first.equals(y.get(i)))
                return false;
        }
        return true;
    }

    private int[] getRandomFeatureIndices(int nFeatures, int mFeatures) {
        List<Integer> all = new ArrayList<>();
        for (int i = 0; i < nFeatures; i++)
            all.add(i);
        Collections.shuffle(all, random);
        mFeatures = Math.min(mFeatures, nFeatures);
        int[] res = new int[mFeatures];
        for (int i = 0; i < mFeatures; i++) {
            res[i] = all.get(i);
        }
        return res;
    }

    private double giniImpurity(List<String> leftY, List<String> rightY) {
        int total = leftY.size() + rightY.size();
        double gLeft = giniOf(leftY);
        double gRight = giniOf(rightY);
        return (leftY.size() * gLeft + rightY.size() * gRight) / total;
    }

    private double giniOf(List<String> y) {
        if (y.isEmpty())
            return 0.0;
        Map<String, Integer> counts = new HashMap<>();
        for (String label : y) {
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }
        double sum = 0.0;
        for (int c : counts.values()) {
            double p = (double) c / y.size();
            sum += p * p;
        }
        return 1.0 - sum;
    }

    private String predictTree(TreeNode node, double[] x) {
        if (node.isLeaf || node.left == null || node.right == null) {
            return node.label;
        }
        if (x[node.featureIndex] <= node.threshold) {
            return predictTree(node.left, x);
        } else {
            return predictTree(node.right, x);
        }
    }

    private static Map<String, Map<String, Double>> calculateMetrics(
            List<String> yActual, List<String> yPredicted, List<String> classLabels) {

        int nClasses = classLabels.size();
        // Confusion Matrix: [Actual Index][Predicted Index] = Count
        int[][] cm = new int[nClasses][nClasses];

        // Map nhãn String sang Index int (0, 1, 2, ...)
        Map<String, Integer> labelToIndex = new HashMap<>();
        for (int i = 0; i < nClasses; i++) {
            labelToIndex.put(classLabels.get(i), i);
        }

        // 1. Confusion Matrix (CM)
        for (int i = 0; i < yActual.size(); i++) {
            String actual = yActual.get(i);
            String predicted = yPredicted.get(i);

            Integer actIndex = labelToIndex.get(actual);
            Integer predIndex = labelToIndex.get(predicted);

            if (actIndex != null && predIndex != null) {
                cm[actIndex][predIndex]++;
            }
        }

        Map<String, Map<String, Double>> metrics = new LinkedHashMap<>();

        double totalF1 = 0;
        double totalPrecision = 0;
        double totalRecall = 0;
        int totalSupport = yActual.size();

        // 2. Tính Precision, Recall, F1
        for (int i = 0; i < nClasses; i++) {
            String label = classLabels.get(i);

            // True Positives (TP) = cm[i][i]
            int tp = cm[i][i];

            // False Positives (FP) = Tổng cột i - TP
            int fp = 0;
            for (int k = 0; k < nClasses; k++) {
                if (k != i)
                    fp += cm[k][i];
            }

            // False Negatives (FN) = Tổng hàng i - TP
            int fn = 0;
            for (int k = 0; k < nClasses; k++) {
                if (k != i)
                    fn += cm[i][k];
            }

            // Support (Tổng số mẫu thực tế của lớp i)
            int support = tp + fn;

            double precision = (tp + fp) == 0 ? 0.0 : (double) tp / (tp + fp);
            double recall = (tp + fn) == 0 ? 0.0 : (double) tp / (tp + fn);
            double f1Score = (precision + recall) == 0 ? 0.0 : 2 * precision * recall / (precision + recall);

            Map<String, Double> classMetrics = new HashMap<>();
            classMetrics.put("Precision", precision);
            classMetrics.put("Recall", recall);
            classMetrics.put("F1-Score", f1Score);
            classMetrics.put("Support", (double) support);
            metrics.put(label, classMetrics);

            totalF1 += f1Score;
            totalPrecision += precision;
            totalRecall += recall;
        }

        // 3. Tính Macro Average (Trung bình cộng)
        Map<String, Double> macroMetrics = new HashMap<>();
        macroMetrics.put("Precision", totalPrecision / nClasses);
        macroMetrics.put("Recall", totalRecall / nClasses);
        macroMetrics.put("F1-Score", totalF1 / nClasses);
        macroMetrics.put("Support", (double) totalSupport);
        metrics.put("Macro Avg", macroMetrics);

        // 4. Lưu Confusion Matrix dưới dạng một Map riêng biệt
        Map<String, Double> cmData = new HashMap<>();
        for (int i = 0; i < nClasses; i++) {
            for (int j = 0; j < nClasses; j++) {
                cmData.put(classLabels.get(i) + "->" + classLabels.get(j), (double) cm[i][j]);
            }
        }
        metrics.put("Confusion Matrix", cmData);

        return metrics;
    }

    /**
     * Trả về danh sách các nhãn duy nhất từ dữ liệu.
     */
    private static List<String> getUniqueLabels(List<String> y) {
        Set<String> unique = new HashSet<>(y);
        List<String> sortedLabels = new ArrayList<>(unique);
        Collections.sort(sortedLabels); // Sắp xếp theo thứ tự để đảm bảo CM ổn định (Low, Medium, High)
        return sortedLabels;
    }
}