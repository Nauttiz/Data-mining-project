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
            String[] values = line.split(",", -1);  // -1 to keep trailing empty strings
            
            if (isHeader) {
                headers = values;
                isHeader = false;
            } else {
                // Ensure all rows have same number of columns as headers
                if (values.length < headers.length) {
                    String[] paddedRow = new String[headers.length];
                    System.arraycopy(values, 0, paddedRow, 0, values.length);
                    for (int i = values.length; i < headers.length; i++) {
                        paddedRow[i] = "";  // Fill missing columns with empty string
                    }
                    data.add(paddedRow);
                } else {
                    data.add(values);
                }
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
            if (columnIndex >= row.length) continue;
            try {
                String val = row[columnIndex];
                if (val != null && !val.trim().isEmpty()) {
                    values.add(Double.parseDouble(val.trim()));
                }
            } catch (NumberFormatException e) {
                // Skip non-numeric values
            }
        }
        
        if (values.isEmpty()) {
            System.out.println("No numeric values found in column " + columnIndex);
            return;
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
            if (columnIndex >= row.length) {
                cleanedData.add(row);
                continue;
            }
            try {
                String val = row[columnIndex];
                if (val == null || val.trim().isEmpty()) {
                    cleanedData.add(row);
                    continue;
                }
                double value = Double.parseDouble(val.trim());
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
            if (columnIndex >= row.length) continue;
            try {
                String val = row[columnIndex];
                if (val != null && !val.trim().isEmpty()) {
                    values.add(Double.parseDouble(val.trim()));
                }
            } catch (NumberFormatException e) {
                // Skip non-numeric values
            }
        }
        
        if (values.isEmpty()) {
            System.out.println("No numeric values found in column " + columnIndex);
            return;
        }
        
        double min = values.stream().mapToDouble(Double::doubleValue).min().orElse(0);
        double max = values.stream().mapToDouble(Double::doubleValue).max().orElse(1);
        
        if (max == min) {
            System.out.println("Column " + columnIndex + " has constant value, skipping normalization");
            return;
        }
        
        for (String[] row : data) {
            if (columnIndex >= row.length) continue;
            try {
                String val = row[columnIndex];
                if (val != null && !val.trim().isEmpty()) {
                    double value = Double.parseDouble(val.trim());
                    double normalized = (value - min) / (max - min);
                    row[columnIndex] = String.format("%.4f", normalized);
                }
            } catch (NumberFormatException e) {
                // Keep original
            }
        }
        
        System.out.println("Normalized column " + columnIndex);
    }

    public void imputeNumericMean(int columnIndex) {
        // Calculate mean from non-missing values
        List<Double> values = new ArrayList<>();
        for (String[] row : data) {
            if (columnIndex >= row.length) continue;
            String val = row[columnIndex];
            if (val != null && !val.trim().isEmpty()) {
                try {
                    values.add(Double.parseDouble(val.trim()));
                } catch (NumberFormatException e) {
                    // Skip non-numeric values
                }
            }
        }
        
        if (values.isEmpty()) {
            System.out.println("No numeric values found in column " + columnIndex);
            return;
        }
        
        double mean = values.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        int imputed = 0;
        
        // Replace missing values with mean
        for (String[] row : data) {
            if (columnIndex >= row.length) continue;
            if (row[columnIndex] == null || row[columnIndex].trim().isEmpty()) {
                row[columnIndex] = String.format("%.4f", mean);
                imputed++;
            }
        }
        
        System.out.println("Imputed " + imputed + " missing values in column " + columnIndex + 
                           " (" + headers[columnIndex] + ") with mean: " + String.format("%.4f", mean));
    }
    
    public Map<String, Integer> labelEncoding(int columnIndex) {
        Map<String, Integer> labelMap = new LinkedHashMap<>();
        int labelCounter = 0;
        
        // Build label map
        for (String[] row : data) {
            if (columnIndex >= row.length) continue;
            String value = row[columnIndex];
            if (value != null) {
                value = value.trim();
                if (!value.isEmpty() && !labelMap.containsKey(value)) {
                    labelMap.put(value, labelCounter++);
                }
            }
        }
        
        // Apply encoding
        for (String[] row : data) {
            if (columnIndex >= row.length) continue;
            String value = row[columnIndex];
            if (value != null) {
                value = value.trim();
                if (labelMap.containsKey(value)) {
                    row[columnIndex] = String.valueOf(labelMap.get(value));
                }
            }
        }
        
        System.out.println("Label encoded column " + columnIndex + " (" + headers[columnIndex] + 
                           ") with " + labelMap.size() + " unique labels");
        
        return labelMap;
    }

    public void discretizeColumn(int columnIndex, double[] cutoffs, String[] labels) {
        if (cutoffs.length != labels.length - 1) {
            System.err.println("Lỗi: Số lượng cutoffs phải bằng số lượng labels trừ 1.");
            return;
        }

        int changed = 0;

        for (String[] row : data) {
            if (columnIndex >= row.length)
                continue;

            try {
                String val = row[columnIndex];
                if (val == null || val.trim().isEmpty()) {
                    continue;
                }

                double value = Double.parseDouble(val.trim());
                String newLabel = labels[labels.length - 1]; 

                for (int i = 0; i < cutoffs.length; i++) {
                    if (value <= cutoffs[i]) {
                        newLabel = labels[i];
                        break;
                    }
                }

          
                row[columnIndex] = newLabel;
                changed++;

            } catch (NumberFormatException e) {
            }
        }

        System.out.println("Discretized " + changed + " values in column " + columnIndex +
                " (" + headers[columnIndex] + ") into " + labels.length + " classes.");

    }

    public void saveCleanedData(String outputPath) throws IOException {
        File file = new File(outputPath);
        // Create parent directories if they don't exist
        if (file.getParentFile() != null) {
            file.getParentFile().mkdirs();
        }
        
        BufferedWriter bw = new BufferedWriter(new FileWriter(file));
        
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