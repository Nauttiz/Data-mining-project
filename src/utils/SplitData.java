package utils;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class SplitData {

    public static void split(String inputPath) throws IOException {
        System.out.println("\n--- Starting Data Split (80/20) ---");
        System.out.println("Reading data from: " + inputPath);

        List<String> lines = Files.readAllLines(Paths.get(inputPath));
        lines.removeIf(s -> s == null || s.trim().isEmpty());

        if (lines.isEmpty()) {
            System.out.println("Error: Input file is empty.");
            return;
        }

        String header = lines.get(0);
        List<String> dataLines = new ArrayList<>(lines.subList(1, lines.size()));

        if (dataLines.isEmpty()) {
            System.out.println("Error: No data rows found.");
            return;
        }

        Collections.shuffle(dataLines, new Random(42));

        int splitIndex = (int) (dataLines.size() * 0.8);

        List<String> trainData = new ArrayList<>();
        List<String> testData = new ArrayList<>();

        trainData.add(header);
        testData.add(header);

        trainData.addAll(dataLines.subList(0, splitIndex));
        testData.addAll(dataLines.subList(splitIndex, dataLines.size()));

        String trainOutputPath = "src/data/train.csv";
        String testOutputPath = "src/data/test.csv";

        Files.write(Paths.get(trainOutputPath), trainData);
        Files.write(Paths.get(testOutputPath), testData);

        System.out.println("--- Data Split Completed ---");
        System.out.println("Total rows: " + dataLines.size());
        System.out.println("Train rows: " + (trainData.size() - 1));
        System.out.println("Test rows: " + (testData.size() - 1));
        System.out.println("Saved to: " + trainOutputPath + " and " + testOutputPath);
    }
}
