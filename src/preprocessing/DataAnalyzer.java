package preprocessing;

import java.util.*;

public class DataAnalyzer {
    private List<String[]> data;
    private String[] headers;
    
    public DataAnalyzer(List<String[]> data, String[] headers) {
        this.data = data;
        this.headers = headers;
    }
    
    public void printStatistics() {
        System.out.println("\n========== DATA STATISTICS ==========");
        System.out.println("Total Records: " + data.size());
        System.out.println("Total Columns: " + headers.length);
        System.out.println("\nColumn Names:");
        for (int i = 0; i < headers.length; i++) {
            System.out.println(i + ": " + headers[i]);
        }
        
        System.out.println("\n========== NUMERIC COLUMNS ANALYSIS ==========");
        for (int i = 0; i < headers.length; i++) {
            analyzeColumn(i);
        }
    }
    
    private void analyzeColumn(int columnIndex) {
        List<Double> values = new ArrayList<>();
        int missingCount = 0;
        
        for (String[] row : data) {
            try {
                if (row[columnIndex] == null || row[columnIndex].trim().isEmpty()) {
                    missingCount++;
                } else {
                    values.add(Double.parseDouble(row[columnIndex]));
                }
            } catch (NumberFormatException e) {
                // Not a numeric column
                return;
            }
        }
        
        if (values.isEmpty()) return;
        
        double[] sorted = values.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        double mean = values.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double min = sorted[0];
        double max = sorted[sorted.length - 1];
        double median = sorted.length % 2 == 0 
            ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2 
            : sorted[sorted.length / 2];
        
        System.out.println("\n" + headers[columnIndex] + ":");
        System.out.println("  Count: " + values.size() + " | Missing: " + missingCount);
        System.out.println("  Mean: " + String.format("%.4f", mean));
        System.out.println("  Median: " + String.format("%.4f", median));
        System.out.println("  Min: " + String.format("%.4f", min));
        System.out.println("  Max: " + String.format("%.4f", max));
    }
    
    public void printDataSample(int numRows) {
        System.out.println("\n========== DATA SAMPLE (First " + numRows + " rows) ==========");
        System.out.println(String.join("\t", headers));
        
        for (int i = 0; i < Math.min(numRows, data.size()); i++) {
            System.out.println(String.join("\t", data.get(i)));
        }
    }
}