
import preprocessing.DataCleaner;
import preprocessing.DataAnalyzer;
import java.io.IOException;

public class Main {
    public static void main(String[] args) {
        try {
            // Step 1: Load and clean the data
            String inputPath = "src/data/World Happiness Report 2024.csv";
            String outputPath = "src/data/cleaned_world_happiness.csv";

            DataCleaner cleaner = new DataCleaner(inputPath);

            // Step 2: Handle missing values
            cleaner.removeMissingValues();

            // Step 3: Remove outliers from Life Ladder column (index 2)
            cleaner.removeOutliers(2, 2.5);

            // Step 4: Normalize numeric columns
            cleaner.normalizeColumn(2);  // Life Ladder
            cleaner.normalizeColumn(3);  // Log GDP
            cleaner.normalizeColumn(6);  // Social Support

            // Step 5: Analyze the cleaned data
            DataAnalyzer analyzer = new DataAnalyzer(
                cleaner.getData(),
                cleaner.getHeaders()
            );

            analyzer.printStatistics();
            analyzer.printDataSample(5);

            // Step 6: Save cleaned data
            cleaner.saveCleanedData(outputPath);

        } catch (IOException e) {
            System.err.println("Error during data processing: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
