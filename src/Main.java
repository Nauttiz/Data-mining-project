import preprocessing.DataCleaner;
import preprocessing.DataAnalyzer;
import Model.RandomForestClassifier;
import Model.PCA_RandomForest;

import utils.SplitData;

public class Main {
    public static void main(String[] args) {
        try {
            String inputPath = "src/data/World Happiness Report 2024.csv";
            String outputPath = "src/data/cleaned_world_happiness.csv";
            String trainPath = "src/data/train.csv";
            String testPath = "src/data/test.csv";

            DataCleaner cleaner = new DataCleaner(inputPath);

            // Impute missing numeric values with mean (all numeric columns)
            cleaner.imputeNumericMean(3); // Log GDP per capita
            cleaner.imputeNumericMean(4); // Social support
            cleaner.imputeNumericMean(5); // Healthy life expectancy at birth
            cleaner.imputeNumericMean(6); // Freedom to make life choices
            cleaner.imputeNumericMean(7); // Generosity
            cleaner.imputeNumericMean(8); // Perceptions of corruption
            cleaner.imputeNumericMean(9); // Positive affect
            cleaner.imputeNumericMean(10); // Negative affect

            // Remove outliers from Life Ladder column (index 2)
            cleaner.removeOutliers(2, 2.5);

            int lifeLadderIndex = 2;

            double[] thresholds = { 5.5, 7.0 };
            String[] classLabels = { "Low", "Medium", "High" };
            cleaner.discretizeColumn(lifeLadderIndex, thresholds, classLabels);

            // Normalize numeric columns
            // cleaner.normalizeColumn(2); // Life Ladder
            cleaner.normalizeColumn(3); // Log GDP per capita
            cleaner.normalizeColumn(4); // Social support
            cleaner.normalizeColumn(5); // Healthy life expectancy at birth
            cleaner.normalizeColumn(6); // Freedom to make life choices
            cleaner.normalizeColumn(7); // Generosity
            cleaner.normalizeColumn(8); // Perceptions of corruption
            cleaner.normalizeColumn(9); // Positive affect
            cleaner.normalizeColumn(10); // Negative affect

            // Analyze the cleaned data
            DataAnalyzer analyzer = new DataAnalyzer(
                    cleaner.getData(),
                    cleaner.getHeaders());

            analyzer.printStatistics();
            analyzer.printDataSample(5);

            SplitData.split(outputPath);

            // Save cleaned data
            cleaner.saveCleanedData(outputPath);

            RandomForestClassifier.runRandomForestTrainTest(trainPath, testPath);

            PCA_RandomForest.runPCAPipeline(trainPath, testPath, 5);

        } catch (Exception e) {
            System.err.println("Error during data processing: " + e.getMessage());
            e.printStackTrace();
        }
    }
}