package Model;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;

import java.util.function.*;
import java.util.stream.*;

public class RandomForestClassifier {

    // ===========================================================
    // HELPER METHODS FOR STRING REPETITION (Java 8 compatible)
    // ===========================================================
    private static String repeatChar(char ch, int count) {
        char[] chars = new char[count];
        Arrays.fill(chars, ch);
        return new String(chars);
    }
    
    private static String repeatString(String str, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(str);
        }
        return sb.toString();
    }

    // ===========================================================
    // TREE NODE CLASS
    // ===========================================================
    private static class TreeNode {
        boolean isLeaf;
        String label;
        int featureIndex;
        double threshold;
        TreeNode left;
        TreeNode right;
    }

    // ===========================================================
    // FIELDS
    // ===========================================================
    private final List<TreeNode> trees;
    private final int numTrees;
    private final int maxDepth;
    private final int minSamplesSplit;
    private final int maxFeatures;
    private final Random random;

    // ===========================================================
    // CONSTRUCTOR
    // ===========================================================
    public RandomForestClassifier(int numTrees, int maxDepth,
                                  int minSamplesSplit, int maxFeatures) {
        this.numTrees = numTrees;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.maxFeatures = maxFeatures;
        this.trees = new ArrayList<>();
        this.random = new Random(42);
    }

    // ===========================================================
    // TRAINING METHODS
    // ===========================================================
    public void fit(List<double[]> X, List<String> y) {
        validateInput(X, y);
        trees.clear();
        
        int nSamples = X.size();
        int nFeatures = X.get(0).length;
        int mFeatures = calculateFeaturesPerTree(nFeatures);
        
        System.out.println("Training Random Forest with " + numTrees + " trees...");
        
        for (int t = 0; t < numTrees; t++) {
            // Bootstrap sampling with replacement
            List<double[]> sampleX = new ArrayList<>();
            List<String> sampleY = new ArrayList<>();
            
            for (int i = 0; i < nSamples; i++) {
                int idx = random.nextInt(nSamples);
                sampleX.add(X.get(idx));
                sampleY.add(y.get(idx));
            }
            
            TreeNode root = buildTree(sampleX, sampleY, 0, mFeatures);
            trees.add(root);
            
            // Progress tracking
            if ((t + 1) % 10 == 0 || t == numTrees - 1) {
                System.out.println("  Trained " + (t + 1) + "/" + numTrees + " trees");
            }
        }
    }
    
    private void validateInput(List<double[]> X, List<String> y) {
        Objects.requireNonNull(X, "Feature matrix cannot be null");
        Objects.requireNonNull(y, "Labels cannot be null");
        
        if (X.isEmpty() || X.size() != y.size()) {
            throw new IllegalArgumentException("Invalid input data dimensions");
        }
    }
    
    private int calculateFeaturesPerTree(int nFeatures) {
        return maxFeatures > 0 ? 
               Math.min(maxFeatures, nFeatures) : 
               (int) Math.sqrt(nFeatures);
    }

    // ===========================================================
    // PREDICTION METHODS
    // ===========================================================
    public String predict(double[] x) {
        if (trees.isEmpty()) {
            throw new IllegalStateException("Model has not been trained");
        }
        
        // Collect votes from all trees
        Map<String, Long> votes = trees.stream()
            .map(tree -> predictTree(tree, x))
            .collect(Collectors.groupingBy(
                Function.identity(),
                Collectors.counting()
            ));
        
        // Return label with most votes
        return votes.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse("Unknown");
    }
    
    public List<String> predict(List<double[]> X) {
        return X.stream()
               .map(this::predict)
               .collect(Collectors.toList());
    }
    
    private String predictTree(TreeNode node, double[] x) {
        while (!node.isLeaf) {
            node = (x[node.featureIndex] <= node.threshold) ? node.left : node.right;
        }
        return node.label;
    }

    // ===========================================================
    // TREE BUILDING METHODS
    // ===========================================================
    private TreeNode buildTree(List<double[]> X, List<String> y, int depth, int mFeatures) {
        TreeNode node = new TreeNode();
        
        // Check stopping conditions
        if (shouldStopBuilding(X, y, depth)) {
            node.isLeaf = true;
            node.label = getMajorityLabel(y);
            return node;
        }
        
        // Find best split
        SplitInfo bestSplit = findBestSplit(X, y, mFeatures);
        
        if (!bestSplit.isValid()) {
            node.isLeaf = true;
            node.label = getMajorityLabel(y);
            return node;
        }
        
        // Create split node
        node.isLeaf = false;
        node.featureIndex = bestSplit.featureIndex;
        node.threshold = bestSplit.threshold;
        node.label = getMajorityLabel(y);
        
        // Build subtrees
        List<double[]> leftX = bestSplit.leftIndices.stream()
                              .map(X::get)
                              .collect(Collectors.toList());
        List<String> leftY = bestSplit.leftIndices.stream()
                            .map(y::get)
                            .collect(Collectors.toList());
        
        List<double[]> rightX = bestSplit.rightIndices.stream()
                               .map(X::get)
                               .collect(Collectors.toList());
        List<String> rightY = bestSplit.rightIndices.stream()
                             .map(y::get)
                             .collect(Collectors.toList());
        
        node.left = buildTree(leftX, leftY, depth + 1, mFeatures);
        node.right = buildTree(rightX, rightY, depth + 1, mFeatures);
        
        return node;
    }
    
    private boolean shouldStopBuilding(List<double[]> X, List<String> y, int depth) {
        return X.isEmpty() || 
               y.isEmpty() || 
               depth >= maxDepth || 
               y.size() < minSamplesSplit || 
               isPure(y);
    }
    
    private SplitInfo findBestSplit(List<double[]> X, List<String> y, int mFeatures) {
        int nFeatures = X.get(0).length;
        int[] candidateFeatures = getRandomFeatures(nFeatures, mFeatures);
        
        SplitInfo bestSplit = new SplitInfo();
        
        for (int featureIndex : candidateFeatures) {
            // Get unique values for this feature
            double[] values = X.stream()
                              .mapToDouble(sample -> sample[featureIndex])
                              .distinct()
                              .sorted()
                              .toArray();
            
            // Try potential thresholds
            for (int i = 0; i < values.length - 1; i++) {
                double threshold = (values[i] + values[i + 1]) / 2.0;
                
                SplitInfo currentSplit = evaluateSplit(X, y, featureIndex, threshold);
                
                if (currentSplit.isValid() && currentSplit.gini < bestSplit.gini) {
                    bestSplit.update(currentSplit);
                }
            }
        }
        
        return bestSplit;
    }
    
    private SplitInfo evaluateSplit(List<double[]> X, List<String> y, 
                                    int featureIndex, double threshold) {
        SplitInfo split = new SplitInfo();
        
        // Partition data
        for (int i = 0; i < X.size(); i++) {
            if (X.get(i)[featureIndex] <= threshold) {
                split.leftIndices.add(i);
            } else {
                split.rightIndices.add(i);
            }
        }
        
        // Check if split is valid
        if (split.leftIndices.isEmpty() || split.rightIndices.isEmpty()) {
            return split;
        }
        
        // Calculate Gini impurity
        List<String> leftY = split.leftIndices.stream()
                            .map(y::get)
                            .collect(Collectors.toList());
        List<String> rightY = split.rightIndices.stream()
                             .map(y::get)
                             .collect(Collectors.toList());
        
        split.featureIndex = featureIndex;
        split.threshold = threshold;
        split.gini = calculateWeightedGini(leftY, rightY);
        
        return split;
    }
    
    private class SplitInfo {
        double gini = Double.MAX_VALUE;
        int featureIndex = -1;
        double threshold = 0;
        List<Integer> leftIndices = new ArrayList<>();
        List<Integer> rightIndices = new ArrayList<>();
        
        boolean isValid() {
            return featureIndex >= 0 && !leftIndices.isEmpty() && !rightIndices.isEmpty();
        }
        
        void update(SplitInfo other) {
            this.gini = other.gini;
            this.featureIndex = other.featureIndex;
            this.threshold = other.threshold;
            this.leftIndices = new ArrayList<>(other.leftIndices);
            this.rightIndices = new ArrayList<>(other.rightIndices);
        }
    }

    // ===========================================================
    // HELPER METHODS
    // ===========================================================
    private boolean isPure(List<String> y) {
        return y.stream()
               .distinct()
               .count() <= 1;
    }
    
    private String getMajorityLabel(List<String> y) {
        return y.stream()
               .collect(Collectors.groupingBy(
                   Function.identity(),
                   Collectors.counting()
               ))
               .entrySet().stream()
               .max(Map.Entry.comparingByValue())
               .map(Map.Entry::getKey)
               .orElse("Unknown");
    }
    
    private double calculateGini(List<String> y) {
        if (y.isEmpty()) return 0.0;
        
        Map<String, Long> counts = y.stream()
            .collect(Collectors.groupingBy(
                Function.identity(),
                Collectors.counting()
            ));
        
        double sum = counts.values().stream()
            .mapToDouble(count -> Math.pow((double)count / y.size(), 2))
            .sum();
        
        return 1.0 - sum;
    }
    
    private double calculateWeightedGini(List<String> leftY, List<String> rightY) {
        int total = leftY.size() + rightY.size();
        if (total == 0) return 0.0;
        
        double leftGini = calculateGini(leftY);
        double rightGini = calculateGini(rightY);
        
        return (leftY.size() * leftGini + rightY.size() * rightGini) / total;
    }
    
    private int[] getRandomFeatures(int nTotalFeatures, int nSelectedFeatures) {
        List<Integer> allFeatures = IntStream.range(0, nTotalFeatures)
            .boxed()
            .collect(Collectors.toList());
        
        Collections.shuffle(allFeatures, random);
        
        return allFeatures.stream()
               .limit(nSelectedFeatures)
               .mapToInt(Integer::intValue)
               .toArray();
    }

    // ===========================================================
    // EVALUATION METRICS
    // ===========================================================
    public double calculateAccuracy(List<String> yTrue, List<String> yPred) {
        if (yTrue.size() != yPred.size() || yTrue.isEmpty()) {
            return 0.0;
        }
        
        long correct = IntStream.range(0, yTrue.size())
                      .filter(i -> yTrue.get(i).equals(yPred.get(i)))
                      .count();
        
        return (double) correct / yTrue.size();
    }
    
    public Map<String, Map<String, Double>> calculateClassificationReport(
            List<String> yTrue, List<String> yPred, List<String> labels) {
        
        int[][] confusionMatrix = calculateConfusionMatrix(yTrue, yPred, labels);
        Map<String, Map<String, Double>> report = new HashMap<>();
        
        double[] macroMetrics = {0.0, 0.0, 0.0};
        
        IntStream.range(0, labels.size()).forEach(i -> {
            String label = labels.get(i);
            
            int TP = confusionMatrix[i][i];
            
            int FN = IntStream.range(0, labels.size())
                             .filter(j -> j != i)
                             .map(j -> confusionMatrix[i][j])
                             .sum();
            
            int FP = IntStream.range(0, labels.size())
                             .filter(j -> j != i)
                             .map(j -> confusionMatrix[j][i])
                             .sum();
            
            int support = Arrays.stream(confusionMatrix[i]).sum();
            
            double precision = (TP + FP == 0) ? 0.0 : (double) TP / (TP + FP);
            double recall = (TP + FN == 0) ? 0.0 : (double) TP / (TP + FN);
            double f1 = (precision + recall == 0) ? 0.0 : 
                        2 * precision * recall / (precision + recall);
            
            Map<String, Double> metrics = new HashMap<>();
            metrics.put("precision", precision);
            metrics.put("recall", recall);
            metrics.put("f1", f1);
            metrics.put("support", (double) support);
            
            report.put(label, metrics);
            
            macroMetrics[0] += precision;
            macroMetrics[1] += recall;
            macroMetrics[2] += f1;
        });
        
        // Calculate macro averages
        int numClasses = labels.size();
        Map<String, Double> macroAvg = new HashMap<>();
        macroAvg.put("precision", macroMetrics[0] / numClasses);
        macroAvg.put("recall", macroMetrics[1] / numClasses);
        macroAvg.put("f1", macroMetrics[2] / numClasses);
        macroAvg.put("support", (double) yTrue.size());
        
        report.put("macro_avg", macroAvg);
        
        return report;
    }
    
    public int[][] calculateConfusionMatrix(List<String> yTrue, 
                                            List<String> yPred, 
                                            List<String> labels) {
        int n = labels.size();
        int[][] matrix = new int[n][n];
        
        Map<String, Integer> labelIndex = IntStream.range(0, n)
            .boxed()
            .collect(Collectors.toMap(
                labels::get,
                Function.identity()
            ));
        
        IntStream.range(0, yTrue.size())
            .forEach(i -> {
                int actual = labelIndex.get(yTrue.get(i));
                int predicted = labelIndex.get(yPred.get(i));
                matrix[actual][predicted]++;
            });
        
        return matrix;
    }

    // ===========================================================
    // DATA LOADING METHODS
    // ===========================================================
    private static void loadData(String filePath, 
                                 List<double[]> features, 
                                 List<String> labels) throws IOException {
        
        if (!Files.exists(Paths.get(filePath))) {
            throw new IOException("File does not exist: " + filePath);
        }
        
        List<String> lines = Files.readAllLines(Paths.get(filePath));
        if (lines.isEmpty()) {
            throw new IOException("File is empty: " + filePath);
        }
        
        // Skip header and process each line
        lines.stream()
            .skip(1)
            .filter(line -> !line.trim().isEmpty())
            .forEach(line -> {
                String[] parts = line.split(",", -1);
                
                if (parts.length >= 3) {
                    try {
                        String label = parts[2].trim();
                        
                        double[] featureArray = new double[parts.length - 2];
                        
                        // First feature from column 1
                        featureArray[0] = parseDoubleSafe(parts[1]);
                        
                        // Remaining features from columns 3 onward
                        for (int j = 3; j < parts.length; j++) {
                            featureArray[j - 2] = parseDoubleSafe(parts[j]);
                        }
                        
                        labels.add(label);
                        features.add(featureArray);
                    } catch (NumberFormatException e) {
                        System.err.println("Skipping malformed line: " + line);
                    }
                }
            });
        
        System.out.println("Loaded " + features.size() + " samples from: " + filePath);
    }
    
    private static double parseDoubleSafe(String value) {
        try {
            return value == null || value.trim().isEmpty() ? 
                   0.0 : Double.parseDouble(value.trim());
        } catch (NumberFormatException e) {
            return 0.0;
        }
    }
    
    private static List<String> getUniqueLabels(List<String> labels) {
        return labels.stream()
               .distinct()
               .sorted()
               .collect(Collectors.toList());
    }

    // ===========================================================
    // MAIN TRAIN-TEST PIPELINE (Java 8 compatible)
    // ===========================================================
    public static void runRandomForestTrainTest(String trainPath, String testPath) {
        try {
            System.out.println(repeatChar('=', 80));
            System.out.println("RANDOM FOREST CLASSIFIER - TRAIN/TEST EVALUATION");
            System.out.println(repeatChar('=', 80));
            
            // Load training data
            System.out.println("\n[1] Loading training data...");
            List<double[]> X_train = new ArrayList<>();
            List<String> y_train = new ArrayList<>();
            loadData(trainPath, X_train, y_train);
            
            // Load test data
            System.out.println("[2] Loading test data...");
            List<double[]> X_test = new ArrayList<>();
            List<String> y_test = new ArrayList<>();
            loadData(testPath, X_test, y_test);
            
            if (X_train.isEmpty() || X_test.isEmpty()) {
                System.err.println("Error: Empty dataset!");
                return;
            }
            
            // Set hyperparameters
            int numTrees = 50;
            int maxDepth = 10;
            int minSamplesSplit = 2;
            int nFeatures = X_train.get(0).length;
            int maxFeatures = Math.max(3, (int) Math.sqrt(nFeatures));
            
            System.out.println("\n[3] Hyperparameters:");
            System.out.println("    - Number of trees:      " + numTrees);
            System.out.println("    - Maximum depth:        " + maxDepth);
            System.out.println("    - Min samples split:    " + minSamplesSplit);
            System.out.println("    - Max features:         " + maxFeatures);
            System.out.println("    - Total features:       " + nFeatures);
            
            // Train model
            System.out.println("\n[4] Training Random Forest...");
            long trainStart = System.currentTimeMillis();
            
            RandomForestClassifier model = new RandomForestClassifier(
                numTrees, maxDepth, minSamplesSplit, maxFeatures);
            model.fit(X_train, y_train);
            
            long trainEnd = System.currentTimeMillis();
            long trainTime = trainEnd - trainStart;
            System.out.println("    Training completed in " + trainTime + " ms");
            
            // Make predictions
            System.out.println("\n[5] Making predictions...");
            long predictStart = System.currentTimeMillis();
            
            List<String> y_pred = model.predict(X_test);
            double accuracy = model.calculateAccuracy(y_test, y_pred);
            
            long predictEnd = System.currentTimeMillis();
            long predictTime = predictEnd - predictStart;
            System.out.println("    Prediction completed in " + predictTime + " ms");
            
            // Get unique labels
            List<String> uniqueLabels = getUniqueLabels(y_train);
            
            // Calculate metrics
            System.out.println("\n[6] Calculating metrics...");
            Map<String, Map<String, Double>> report = 
                model.calculateClassificationReport(y_test, y_pred, uniqueLabels);
            
            int[][] confusionMatrix = 
                model.calculateConfusionMatrix(y_test, y_pred, uniqueLabels);
            
            // Print results
            printResultsSummary(X_train, X_test, uniqueLabels, 
                                trainTime, predictTime, accuracy);
            
            printClassificationReport(report, uniqueLabels);
            
            printConfusionMatrix(confusionMatrix, uniqueLabels);
            
            printSamplePredictions(y_test, y_pred);
            
            System.out.println("\n" + repeatChar('=', 80));
            System.out.println("EVALUATION COMPLETE");
            System.out.println(repeatChar('=', 80));
            
        } catch (Exception e) {
            System.err.println("\nERROR: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void printResultsSummary(List<double[]> X_train, List<double[]> X_test,
                                            List<String> labels, long trainTime, 
                                            long predictTime, double accuracy) {
        System.out.println("\n" + repeatChar('=', 80));
        System.out.println("RESULTS SUMMARY");
        System.out.println(repeatChar('=', 80));
        
        System.out.println("\nDATASET INFORMATION:");
        System.out.printf("  Training samples:  %10d\n", X_train.size());
        System.out.printf("  Test samples:      %10d\n", X_test.size());
        System.out.printf("  Number of classes: %10d\n", labels.size());
        
        System.out.println("\nPERFORMANCE TIMING:");
        System.out.printf("  Training time:     %10d ms\n", trainTime);
        System.out.printf("  Prediction time:   %10d ms\n", predictTime);
        System.out.printf("  Total time:        %10d ms\n", trainTime + predictTime);
        
        System.out.println("\nMODEL PERFORMANCE:");
        System.out.printf("  Accuracy:          %10.4f (%.2f%%)\n", 
                         accuracy, accuracy * 100);
    }
    
    private static void printClassificationReport(Map<String, Map<String, Double>> report,
                                                  List<String> labels) {
        System.out.println("\n" + repeatChar('-', 80));
        System.out.println("CLASSIFICATION REPORT");
        System.out.println(repeatChar('-', 80));
        
        System.out.printf("\n%-15s %-12s %-12s %-12s %-12s\n", 
            "Class", "Precision", "Recall", "F1-Score", "Support");
        System.out.println(repeatChar('-', 75));
        
        labels.forEach(label -> {
            Map<String, Double> metrics = report.get(label);
            System.out.printf("%-15s %-12.4f %-12.4f %-12.4f %-12.0f\n",
                label,
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1"),
                metrics.get("support"));
        });
        
        Map<String, Double> macroAvg = report.get("macro_avg");
        System.out.println(repeatChar('-', 75));
        System.out.printf("%-15s %-12.4f %-12.4f %-12.4f %-12.0f\n",
            "Macro Avg",
            macroAvg.get("precision"),
            macroAvg.get("recall"),
            macroAvg.get("f1"),
            macroAvg.get("support"));
    }
    
    private static void printConfusionMatrix(int[][] matrix, List<String> labels) {
        System.out.println("\n" + repeatChar('-', 80));
        System.out.println("CONFUSION MATRIX");
        System.out.println(repeatChar('-', 80));
        
        System.out.printf("\n%-15s", "Actual \\ Predicted");
        labels.forEach(label -> System.out.printf(" %-10s", label));
        System.out.println();
        
        System.out.println(repeatChar('-', 15 + 11 * labels.size()));
        
        IntStream.range(0, labels.size())
                .forEach(i -> {
                    System.out.printf("%-15s", labels.get(i));
                    IntStream.range(0, labels.size())
                            .forEach(j -> System.out.printf(" %-10d", matrix[i][j]));
                    System.out.println();
                });
    }
    
    private static void printSamplePredictions(List<String> yTrue, List<String> yPred) {
        System.out.println("\n" + repeatChar('-', 80));
        System.out.println(" PREDICTIONS (First 10 instances)");
        System.out.println(repeatChar('-', 80));
        
        System.out.printf("\n%-8s %-15s %-15s %-10s\n", 
            "Index", "Actual", "Predicted", "Result");
        System.out.println(repeatChar('-', 50));
        
        int limit = Math.min(10, yTrue.size());
        IntStream.range(0, limit)
                .forEach(i -> {
                    boolean correct = yTrue.get(i).equals(yPred.get(i));
                    String result = correct ? " CORRECT" : " WRONG";
                    System.out.printf("%-8d %-15s %-15s %-10s\n",
                        i + 1, yTrue.get(i), yPred.get(i), result);
                });
    }

}