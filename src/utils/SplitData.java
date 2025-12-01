package utils;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class SplitData {

    public static void split(String inputPath) throws IOException {
        // Create data folder if not exists
        File dataFolder = new File("data");
        if (!dataFolder.exists()) {
            dataFolder.mkdirs();
        }

        System.out.println("\n--- Starting Data Split (80/20) ---");
        System.out.println("Reading data from: " + inputPath);
        
        // Read all cleaned data lines
        List<String> lines = Files.readAllLines(Paths.get(inputPath));
        
        // Remove empty or blank lines (Java 8/11 compatible fix)
        lines.removeIf(s -> s == null || s.trim().isEmpty()); 

        if (lines.isEmpty()) {
            System.out.println("Error: Input file is empty or no data was read.");
            return;
        }

        // Separate the Header (assumes the first line is the header)
        String header = lines.get(0);
        List<String> dataLines = lines.subList(1, lines.size());

        if (dataLines.isEmpty()) {
            System.out.println("Error: No data remaining after header separation.");
            return;
        }

        // Shuffle the data lines
        Collections.shuffle(dataLines, new Random());

        // Compute split index (80% train, 20% test)
        int splitIndex = (int) (dataLines.size() * 0.8);

        // FIX for ConcurrentModificationException:
        // Create new ArrayList instances from the subList view for independent modification (copying the data)
        List<String> trainData = new ArrayList<>(dataLines.subList(0, splitIndex));
        List<String> testData = new ArrayList<>(dataLines.subList(splitIndex, dataLines.size()));
        
        // Add the Header to the top of both sets (Now safe to modify)
        trainData.add(0, header);
        testData.add(0, header);

        // Output paths
        String trainOutputPath = "src/data/train.csv";
        String testOutputPath = "src/data/test.csv";
        
        // Write data to files
        Files.write(Paths.get(trainOutputPath), trainData);
        Files.write(Paths.get(testOutputPath), testData);

        System.out.println("--- Data Split Complete! ---");
        System.out.println("Total data rows (excluding header): " + (lines.size() - 1));
        System.out.println("Train size: " + (trainData.size() - 1) + " data rows (+1 header)");
        System.out.println("Test size: " + (testData.size() - 1) + " data rows (+1 header)");
        System.out.println("Saved to: " + trainOutputPath + " and " + testOutputPath);
    }
}