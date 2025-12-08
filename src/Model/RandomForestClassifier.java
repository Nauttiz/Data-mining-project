package Model;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.function.*;
import java.util.stream.*;

public class RandomForestClassifier {

    private static class TreeNode {
        boolean isLeaf;
        String label;
        int featureIndex;
        double threshold;
        TreeNode left;
        TreeNode right;
    }

    private final List<TreeNode> trees;
    private final int numTrees;
    private final int maxDepth;
    private final int minSamplesSplit;
    private final int maxFeatures;
    private final Random random;

    public RandomForestClassifier(int numTrees, int maxDepth,
                                  int minSamplesSplit, int maxFeatures) {
        this.numTrees = numTrees;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.maxFeatures = maxFeatures;
        this.trees = new ArrayList<>();
        this.random = new Random(42);
    }

    public void fit(List<double[]> X, List<String> y) {
        validateInput(X, y);
        trees.clear();
        
        int nSamples = X.size();
        int nFeatures = X.get(0).length;
        int mFeatures = calculateFeaturesPerTree(nFeatures);
        
        System.out.println("Training " + numTrees + " trees...");
        
        for (int t = 0; t < numTrees; t++) {
            // Bootstrap sampling
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
        System.out.println("Training completed");
    }
    
    private void validateInput(List<double[]> X, List<String> y) {
        if (X.isEmpty() || X.size() != y.size()) {
            throw new IllegalArgumentException("Invalid input data dimensions");
        }
    }
    
    private int calculateFeaturesPerTree(int nFeatures) {
        return maxFeatures > 0 ? 
               Math.min(maxFeatures, nFeatures) : 
               (int) Math.sqrt(nFeatures);
    }

    public String predict(double[] x) {
        if (trees.isEmpty()) {
            throw new IllegalStateException("Model has not been trained");
        }
        
        Map<String, Long> votes = trees.stream()
            .map(tree -> predictTree(tree, x))
            .collect(Collectors.groupingBy(
                Function.identity(),
                Collectors.counting()
            ));
        
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

    private TreeNode buildTree(List<double[]> X, List<String> y, int depth, int mFeatures) {
        TreeNode node = new TreeNode();
        
        // Check stopping conditions
        if (X.isEmpty() || depth >= maxDepth || y.size() < minSamplesSplit || isPure(y)) {
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
    
    private SplitInfo findBestSplit(List<double[]> X, List<String> y, int mFeatures) {
        int nFeatures = X.get(0).length;
        int[] candidateFeatures = getRandomFeatures(nFeatures, mFeatures);
        
        SplitInfo bestSplit = new SplitInfo();
        
        for (int featureIndex : candidateFeatures) {
            double[] values = X.stream()
                              .mapToDouble(sample -> sample[featureIndex])
                              .distinct()
                              .sorted()
                              .toArray();
            
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
        
        for (int i = 0; i < X.size(); i++) {
            if (X.get(i)[featureIndex] <= threshold) {
                split.leftIndices.add(i);
            } else {
                split.rightIndices.add(i);
            }
        }
        
        if (split.leftIndices.isEmpty() || split.rightIndices.isEmpty()) {
            return split;
        }
        
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

    private boolean isPure(List<String> y) {
        return y.stream().distinct().count() <= 1;
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
            List<String> yTrue, List<String> yPred) {
        
        List<String> labels = yTrue.stream()
                               .distinct()
                               .sorted()
                               .collect(Collectors.toList());
        
        Map<String, Integer> labelIndex = IntStream.range(0, labels.size())
            .boxed()
            .collect(Collectors.toMap(
                labels::get,
                Function.identity()
            ));
        
        int n = labels.size();
        int[][] confusionMatrix = new int[n][n];
        
        IntStream.range(0, yTrue.size())
            .forEach(i -> {
                int actual = labelIndex.get(yTrue.get(i));
                int predicted = labelIndex.get(yPred.get(i));
                confusionMatrix[actual][predicted]++;
            });
        
        Map<String, Map<String, Double>> report = new HashMap<>();
        double[] macroMetrics = {0.0, 0.0, 0.0};
        
        for (int i = 0; i < labels.size(); i++) {
            String label = labels.get(i);
            
            int TP = confusionMatrix[i][i];
            
            int FN = 0;
            int FP = 0;
            int support = 0;
            
            for (int j = 0; j < labels.size(); j++) {
                support += confusionMatrix[i][j];
                if (j != i) {
                    FN += confusionMatrix[i][j];
                    FP += confusionMatrix[j][i];
                }
            }
            
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
        }
        
        // Macro averages
        int numClasses = labels.size();
        Map<String, Double> macroAvg = new HashMap<>();
        macroAvg.put("precision", macroMetrics[0] / numClasses);
        macroAvg.put("recall", macroMetrics[1] / numClasses);
        macroAvg.put("f1", macroMetrics[2] / numClasses);
        macroAvg.put("support", (double) yTrue.size());
        
        report.put("macro_avg", macroAvg);
        
        return report;
    }

    private static void loadData(String filePath, 
                                 List<double[]> features, 
                                 List<String> labels) throws IOException {
        
        List<String> lines = Files.readAllLines(Paths.get(filePath));
        
        lines.stream()
            .skip(1)
            .filter(line -> !line.trim().isEmpty())
            .forEach(line -> {
                String[] parts = line.split(",", -1);
                
                if (parts.length >= 3) {
                    try {
                        String label = parts[2].trim();
                        double[] featureArray = new double[parts.length - 2];
                        
                        featureArray[0] = parseDoubleSafe(parts[1]);
                        
                        for (int j = 3; j < parts.length; j++) {
                            featureArray[j - 2] = parseDoubleSafe(parts[j]);
                        }
                        
                        labels.add(label);
                        features.add(featureArray);
                    } catch (Exception e) {
                        // Skip malformed lines
                    }
                }
            });
    }
    
    private static double parseDoubleSafe(String value) {
        try {
            return value == null || value.trim().isEmpty() ? 
                   0.0 : Double.parseDouble(value.trim());
        } catch (NumberFormatException e) {
            return 0.0;
        }
    }

    public static void runRandomForestTrainTest(String trainPath, String testPath) {
        try {
            System.out.println("=== Random Forest Classifier ===");
            
            // Load data
            List<double[]> X_train = new ArrayList<>();
            List<String> y_train = new ArrayList<>();
            loadData(trainPath, X_train, y_train);
            
            List<double[]> X_test = new ArrayList<>();
            List<String> y_test = new ArrayList<>();
            loadData(testPath, X_test, y_test);
            
            if (X_train.isEmpty() || X_test.isEmpty()) {
                System.err.println("Error: Empty dataset!");
                return;
            }
            
            // Hyperparameters
            int numTrees = 50;
            int maxDepth = 10;
            int minSamplesSplit = 2;
            int nFeatures = X_train.get(0).length;
            int maxFeatures = Math.max(3, (int) Math.sqrt(nFeatures));
            
            System.out.println("\nHyperparameters:");
            System.out.println("  Trees: " + numTrees + ", Max Depth: " + maxDepth);
            System.out.println("  Min Samples Split: " + minSamplesSplit);
            System.out.println("  Features per tree: " + maxFeatures);
            
            // Train model
            long startTime = System.currentTimeMillis();
            
            RandomForestClassifier model = new RandomForestClassifier(
                numTrees, maxDepth, minSamplesSplit, maxFeatures);
            model.fit(X_train, y_train);
            
            // Predictions
            List<String> y_pred = model.predict(X_test);
            double accuracy = model.calculateAccuracy(y_test, y_pred);
            
            long endTime = System.currentTimeMillis();
            
            // Results
            System.out.println("\nResults:");
            System.out.println("  Training time: " + (endTime - startTime) + " ms");
            System.out.println("  Accuracy: " + String.format("%.2f%%", accuracy * 100));
            
            // Classification report
            Map<String, Map<String, Double>> report = model.calculateClassificationReport(y_test, y_pred);
            
            System.out.println("\nClassification Report:");
            System.out.println("Class       Precision   Recall      F1-Score    Support");
            System.out.println("--------------------------------------------------------");
            
            for (String label : report.keySet()) {
                if (!label.equals("macro_avg")) {
                    Map<String, Double> metrics = report.get(label);
                    System.out.printf("%-10s  %-10.4f  %-10.4f  %-10.4f  %-10.0f%n",
                        label,
                        metrics.get("precision"),
                        metrics.get("recall"),
                        metrics.get("f1"),
                        metrics.get("support"));
                }
            }
            
            // Macro average
            Map<String, Double> macroAvg = report.get("macro_avg");
            System.out.println("--------------------------------------------------------");
            System.out.printf("%-10s  %-10.4f  %-10.4f  %-10.4f  %-10.0f%n",
                "Macro Avg",
                macroAvg.get("precision"),
                macroAvg.get("recall"),
                macroAvg.get("f1"),
                macroAvg.get("support"));
                
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}