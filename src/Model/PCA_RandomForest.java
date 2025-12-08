package Model;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

public class PCA_RandomForest {

    // Load data from CSV
    public static void loadData(String filePath,
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
    }

    private static double parseDoubleSafe(String value) {
        if (value == null || value.trim().isEmpty()) {
            return 0.0;
        }
        try {
            return Double.parseDouble(value.trim());
        } catch (NumberFormatException e) {
            return 0.0;
        }
    }

    // Get unique labels
    private static List<String> getUniqueLabels(List<String> labels) {
        return labels.stream()
                .distinct()
                .sorted()
                .collect(Collectors.toList());
    }

    // Helper method to calculate confusion matrix
    private static int[][] calculateConfusionMatrix(List<String> yTrue, List<String> yPred, List<String> labels) {
        int n = labels.size();
        int[][] matrix = new int[n][n];
        
        Map<String, Integer> labelIndex = new HashMap<>();
        for (int i = 0; i < n; i++) {
            labelIndex.put(labels.get(i), i);
        }
        
        for (int i = 0; i < yTrue.size(); i++) {
            int actual = labelIndex.get(yTrue.get(i));
            int predicted = labelIndex.get(yPred.get(i));
            matrix[actual][predicted]++;
        }
        
        return matrix;
    }

    // Main method to run PCA + RandomForest
    public static void runPCAPipeline(String trainPath, String testPath,
            int nComponents) {
        try {
            System.out.println("=== PCA + Random Forest ===");
            
            // 1. Load original data
            System.out.println("\nLoading data...");
            List<double[]> X_train_orig = new ArrayList<>();
            List<String> y_train = new ArrayList<>();
            loadData(trainPath, X_train_orig, y_train);

            List<double[]> X_test_orig = new ArrayList<>();
            List<String> y_test = new ArrayList<>();
            loadData(testPath, X_test_orig, y_test);

            int originalFeatures = X_train_orig.get(0).length;
            System.out.println("Training samples: " + X_train_orig.size());
            System.out.println("Test samples: " + X_test_orig.size());
            System.out.println("Original features: " + originalFeatures);

            // 2. Apply PCA
            System.out.println("\nApplying PCA...");
            System.out.println("PCA components: " + nComponents);

            PCA pca = new PCA(nComponents);
            List<double[]> X_train_pca = pca.fitTransform(X_train_orig);
            List<double[]> X_test_pca = pca.transform(X_test_orig);

            System.out.println("Features after PCA: " + X_train_pca.get(0).length);

            // 3. Train Random Forest
            System.out.println("\nTraining Random Forest...");
            int maxFeatures = (int) Math.max(1, Math.sqrt(nComponents));
            RandomForestClassifier rfModel = new RandomForestClassifier(
                    50, 10, 2, maxFeatures);

            long trainStart = System.currentTimeMillis();
            rfModel.fit(X_train_pca, y_train);
            long trainTime = System.currentTimeMillis() - trainStart;

            System.out.println("Training time: " + trainTime + " ms");

            // 4. Make predictions
            System.out.println("\nMaking predictions...");
            long predictStart = System.currentTimeMillis();
            List<String> y_pred = rfModel.predict(X_test_pca);
            long predictTime = System.currentTimeMillis() - predictStart;

            double accuracy = rfModel.calculateAccuracy(y_test, y_pred);
            System.out.println("Prediction time: " + predictTime + " ms");
            System.out.printf("Accuracy: %.2f%%\n", accuracy * 100);

            // 5. Get unique labels from combined train and test
            List<String> allLabels = new ArrayList<>(y_train);
            allLabels.addAll(y_test);
            List<String> labels = getUniqueLabels(allLabels);
            
            // 6. Calculate classification report
            System.out.println("\nClassification Report:");
            System.out.println("Class       Precision   Recall      F1-Score    Support");
            System.out.println("--------------------------------------------------------");
            
            // Calculate metrics manually
            int[][] confusionMatrix = calculateConfusionMatrix(y_test, y_pred, labels);
            
            Map<String, Double> precisionMap = new HashMap<>();
            Map<String, Double> recallMap = new HashMap<>();
            Map<String, Double> f1Map = new HashMap<>();
            Map<String, Double> supportMap = new HashMap<>();
            
            double macroPrecision = 0.0;
            double macroRecall = 0.0;
            double macroF1 = 0.0;
            
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
                
                precisionMap.put(label, precision);
                recallMap.put(label, recall);
                f1Map.put(label, f1);
                supportMap.put(label, (double) support);
                
                macroPrecision += precision;
                macroRecall += recall;
                macroF1 += f1;
                
                System.out.printf("%-10s  %-10.4f  %-10.4f  %-10.4f  %-10.0f%n",
                        label, precision, recall, f1, (double) support);
            }
            
            // Macro averages
            int numClasses = labels.size();
            macroPrecision /= numClasses;
            macroRecall /= numClasses;
            macroF1 /= numClasses;
            
            System.out.println("--------------------------------------------------------");
            System.out.printf("%-10s  %-10.4f  %-10.4f  %-10.4f  %-10.0f%n",
                    "Macro Avg", macroPrecision, macroRecall, macroF1, (double) y_test.size());

            // 7. Confusion Matrix (only if small number of classes)
            if (labels.size() <= 10) {
                System.out.println("\nConfusion Matrix:");
                System.out.printf("%-15s", "Actual\\Pred");
                for (String label : labels) {
                    System.out.printf(" %-8s", label);
                }
                System.out.println();
                
                for (int i = 0; i < labels.size(); i++) {
                    System.out.printf("%-15s", labels.get(i));
                    for (int j = 0; j < labels.size(); j++) {
                        System.out.printf(" %-8d", confusionMatrix[i][j]);
                    }
                    System.out.println();
                }
            }

            // 8. Summary
            System.out.println("\nSummary:");
            System.out.println("Original features: " + originalFeatures);
            System.out.println("PCA components: " + nComponents);
            System.out.printf("Accuracy: %.2f%%\n", accuracy * 100);
            System.out.printf("Macro F1-Score: %.4f\n", macroF1);
            System.out.println("Training time: " + trainTime + " ms");
            System.out.println("Prediction time: " + predictTime + " ms");
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}