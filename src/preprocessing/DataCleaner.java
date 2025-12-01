package preprocessing;

import java.io.*;
import java.util.*;

public class DataCleaner {
    private List<String[]> data;
    private String[] headers;
    
    public DataCleaner(String csvFilePath) throws IOException {
        loadCSV(csvFilePath);
    }
    
    private void loadCSV(String filePath) throws IOException {
        data = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line;
        boolean isHeader = true;
        
        while ((line = br.readLine()) != null) {
            String[] values = line.split(",");
            
            if (isHeader) {
                headers = values;
                isHeader = false;
            } else {
                data.add(values);
            }
        }
        br.close();
        System.out.println("Loaded " + data.size() + " records");
    }
    
    public void removeMissingValues() {
        List<String[]> cleanedData = new ArrayList<>();
        int removed = 0;
        
        for (String[] row : data) {
            boolean hasMissing = false;
            for (String value : row) {
                if (value == null || value.trim().isEmpty()) {
                    hasMissing = true;
                    break;
                }
            }
            if (!hasMissing) {
                cleanedData.add(row);
            } else {
                removed++;
            }
        }
        
        data = cleanedData;
        System.out.println("Removed " + removed + " rows with missing values. Remaining: " + data.size());
    }
    
    public void removeOutliers(int columnIndex, double threshold) {
        List<Double> values = new ArrayList<>();
        
        for (String[] row : data) {
            try {
                values.add(Double.parseDouble(row[columnIndex]));
            } catch (NumberFormatException e) {
                // Skip non-numeric values
            }
        }
        
        double mean = values.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double stdDev = Math.sqrt(
            values.stream()
                .mapToDouble(v -> Math.pow(v - mean, 2))
                .average()
                .orElse(0)
        );
        
        List<String[]> cleanedData = new ArrayList<>();
        int removed = 0;
        
        for (String[] row : data) {
            try {
                double value = Double.parseDouble(row[columnIndex]);
                if (Math.abs(value - mean) <= threshold * stdDev) {
                    cleanedData.add(row);
                } else {
                    removed++;
                }
            } catch (NumberFormatException e) {
                cleanedData.add(row);
            }
        }
        
        data = cleanedData;
        System.out.println("Removed " + removed + " outliers. Remaining: " + data.size());
    }
    
    public void normalizeColumn(int columnIndex) {
        List<Double> values = new ArrayList<>();
        
        for (String[] row : data) {
            try {
                values.add(Double.parseDouble(row[columnIndex]));
            } catch (NumberFormatException e) {
                // Skip
            }
        }
        
        double min = values.stream().mapToDouble(Double::doubleValue).min().orElse(0);
        double max = values.stream().mapToDouble(Double::doubleValue).max().orElse(1);
        
        for (String[] row : data) {
            try {
                double value = Double.parseDouble(row[columnIndex]);
                double normalized = (value - min) / (max - min);
                row[columnIndex] = String.format("%.4f", normalized);
            } catch (NumberFormatException e) {
                // Keep original
            }
        }
        
        System.out.println("Normalized column " + columnIndex);
    }
    
    public void saveCleanedData(String outputPath) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(outputPath));
        
        // Write headers
        bw.write(String.join(",", headers));
        bw.newLine();
        
        // Write data
        for (String[] row : data) {
            bw.write(String.join(",", row));
            bw.newLine();
        }
        
        bw.close();
        System.out.println("Cleaned data saved to: " + outputPath);
    }
    
    public List<String[]> getData() {
        return data;
    }
    
    public String[] getHeaders() {
        return headers;
    }
}