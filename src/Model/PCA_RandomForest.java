package Model;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

public class PCA_RandomForest {

    // Helper method to repeat characters (Java 8 compatible)
    private static String repeatChar(char ch, int count) {
        char[] chars = new char[count];
        Arrays.fill(chars, ch);
        return new String(chars);
    }

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

    // Save transformed data to CSV
    public static void saveTransformedData(String outputPath,
            List<double[]> features,
            List<String> labels) throws IOException {
        File file = new File(outputPath);
        file.getParentFile().mkdirs();

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
            // Write header
            int nComponents = features.get(0).length;
            StringBuilder header = new StringBuilder("ID,Class");
            IntStream.range(0, nComponents)
                    .forEach(i -> header.append(",PC").append(i + 1));
            writer.write(header.toString());
            writer.newLine();

            // Write data
            IntStream.range(0, features.size())
                    .forEach(i -> {
                        try {
                            StringBuilder line = new StringBuilder();
                            line.append(i + 1).append(",");
                            line.append(labels.get(i));

                            Arrays.stream(features.get(i))
                                    .forEach(value -> line.append(",").append(String.format("%.6f", value)));

                            writer.write(line.toString());
                            writer.newLine();
                        } catch (IOException e) {
                            System.err.println("Error writing line: " + e.getMessage());
                        }
                    });
        }
        System.out.println("Saved transformed data to: " + outputPath);
    }

    // Get unique labels
    private static List<String> getUniqueLabels(List<String> labels) {
        return labels.stream()
                .distinct()
                .sorted()
                .collect(Collectors.toList());
    }

    // Main method to run PCA + RandomForest
    public static void runPCAPipeline(String trainPath, String testPath,
            int nComponents) {
        try {
            System.out.println(repeatChar('=', 70));
            System.out.println("PCA + RANDOM FOREST MODEL");
            System.out.println(repeatChar('=', 70));

            // 1. Load original data
            System.out.println("\n[1] LOADING DATA");
            System.out.println(repeatChar('-', 40));

            List<double[]> X_train_orig = new ArrayList<>();
            List<String> y_train = new ArrayList<>();
            loadData(trainPath, X_train_orig, y_train);

            List<double[]> X_test_orig = new ArrayList<>();
            List<String> y_test = new ArrayList<>();
            loadData(testPath, X_test_orig, y_test);

            int originalFeatures = X_train_orig.get(0).length;
            System.out.println("Training samples: " + X_train_orig.size());
            System.out.println("Test samples:     " + X_test_orig.size());
            System.out.println("Original features: " + originalFeatures);

            // 2. Apply PCA
            System.out.println("\n[2] APPLYING PCA");
            System.out.println(repeatChar('-', 40));
            System.out.println("Number of PCA components: " + nComponents);

            PCA pca = new PCA(nComponents);
            List<double[]> X_train_pca = pca.fitTransform(X_train_orig);
            List<double[]> X_test_pca = pca.transform(X_test_orig);

            System.out.println("Features after PCA: " + X_train_pca.get(0).length);
            System.out.println("Dimensionality reduction: " +
                    String.format("%.1f%%", (1.0 - (double) nComponents / originalFeatures) * 100));

            // 3. Save transformed data
            System.out.println("\n[3] SAVING TRANSFORMED DATA");
            System.out.println(repeatChar('-', 40));

            String trainPCApath = "src/data/train_pca.csv";
            String testPCApath = "src/data/test_pca.csv";

            saveTransformedData(trainPCApath, X_train_pca, y_train);
            saveTransformedData(testPCApath, X_test_pca, y_test);

            // 4. Train Random Forest
            System.out.println("\n[4] TRAINING RANDOM FOREST");
            System.out.println(repeatChar('-', 40));

            int maxFeatures = (int) Math.max(1, Math.sqrt(nComponents));
            RandomForestClassifier rfModel = new RandomForestClassifier(
                    50, 10, 2, maxFeatures);

            long trainStart = System.currentTimeMillis();
            rfModel.fit(X_train_pca, y_train);
            long trainTime = System.currentTimeMillis() - trainStart;

            System.out.println("Hyperparameters:");
            System.out.println("  - Trees: " + 50);
            System.out.println("  - Max depth: " + 10);
            System.out.println("  - Max features: " + maxFeatures);
            System.out.println("Training time: " + trainTime + " ms");

            // 5. Make predictions
            System.out.println("\n[5] MAKING PREDICTIONS");
            System.out.println(repeatChar('-', 40));

            long predictStart = System.currentTimeMillis();
            List<String> y_pred = rfModel.predict(X_test_pca);
            long predictTime = System.currentTimeMillis() - predictStart;

            double accuracy = rfModel.calculateAccuracy(y_test, y_pred);
            System.out.println("Prediction time: " + predictTime + " ms");
            System.out.printf("Accuracy: %.4f (%.2f%%)\n", accuracy, accuracy * 100);

            // 6. Calculate detailed metrics
            System.out.println("\n[6] DETAILED METRICS");
            System.out.println(repeatChar('-', 40));

            List<String> labels = getUniqueLabels(y_train);
            Map<String, Map<String, Double>> report = rfModel.calculateClassificationReport(y_test, y_pred, labels);

            // Print classification report
            System.out.printf("\n%-15s %-12s %-12s %-12s %-12s\n",
                    "Class", "Precision", "Recall", "F1-Score", "Support");
            System.out.println(repeatChar('-', 65));

            labels.forEach(label -> {
                Map<String, Double> metrics = report.get(label);
                System.out.printf("%-15s %-12.4f %-12.4f %-12.4f %-12.0f\n",
                        label,
                        metrics.get("precision"),
                        metrics.get("recall"),
                        metrics.get("f1"),
                        metrics.get("support"));
            });

            // Print macro averages
            Map<String, Double> macroAvg = report.get("macro_avg");
            System.out.println(repeatChar('-', 65));
            System.out.printf("%-15s %-12.4f %-12.4f %-12.4f %-12.0f\n",
                    "Macro Avg",
                    macroAvg.get("precision"),
                    macroAvg.get("recall"),
                    macroAvg.get("f1"),
                    macroAvg.get("support"));

            // 7. Confusion Matrix
            System.out.println("\n[7] CONFUSION MATRIX");
            System.out.println(repeatChar('-', 40));

            int[][] confusionMatrix = rfModel.calculateConfusionMatrix(y_test, y_pred, labels);

            System.out.printf("\n%-15s", "Actual\\Pred");
            labels.forEach(label -> System.out.printf(" %-10s", label));
            System.out.println();

            IntStream.range(0, labels.size())
                    .forEach(i -> {
                        System.out.printf("%-15s", labels.get(i));
                        IntStream.range(0, labels.size())
                                .forEach(j -> System.out.printf(" %-10d", confusionMatrix[i][j]));
                        System.out.println();
                    });

            // 8. Sample predictions
            System.out.println("\n[8] SAMPLE PREDICTIONS");
            System.out.println(repeatChar('-', 40));

            int numSamples = Math.min(10, y_test.size());
            System.out.printf("\n%-8s %-15s %-15s %-10s\n",
                    "Index", "Actual", "Predicted", "Result");
            System.out.println(repeatChar('-', 50));

            IntStream.range(0, numSamples)
                    .forEach(i -> {
                        boolean correct = y_test.get(i).equals(y_pred.get(i));
                        String result = correct ? "CORRECT" : "WRONG";
                        System.out.printf("%-8d %-15s %-15s %-10s\n",
                                i + 1, y_test.get(i), y_pred.get(i), result);
                    });

            // // 9. Summary
            // System.out.println("\n[9] SUMMARY");
            // System.out.println(repeatChar('-', 40));

            // System.out.println("\nPCA Configuration:");
            // System.out.println("  - Original features: " + originalFeatures);
            // System.out.println("  - PCA components:    " + nComponents);
            // System.out.println("  - Reduction:         " +
            //         String.format("%.1f%%", (1.0 - (double) nComponents / originalFeatures) * 100));

            // System.out.println("\nModel Performance:");
            // System.out.printf("  - Accuracy:          %.4f\n", accuracy);
            // System.out.printf("  - Macro F1-Score:    %.4f\n", macroAvg.get("f1"));
            // System.out.printf("  - Training time:     %d ms\n", trainTime);
            // System.out.printf("  - Prediction time:   %d ms\n", predictTime);

            // System.out.println("\nOutput Files:");
            // System.out.println("  - train_pca.csv:     " + trainPCApath);
            // System.out.println("  - test_pca.csv:      " + testPCApath);

            // System.out.println("\n" + repeatChar('=', 70));
            // System.out.println("PCA + RANDOM FOREST COMPLETED");
            // System.out.println(repeatChar('=', 70));

        } catch (Exception e) {
            System.err.println("\nERROR: " + e.getMessage());
            e.printStackTrace();
        }
    }

}