package preprocessing;

import java.io.*;
import java.util.*;

public class ArffConverter {
    private List<String[]> data;
    private String[] headers;
    
    public ArffConverter(List<String[]> data, String[] headers) {
        this.data = data;
        this.headers = headers;
    }
    
    public void convertToARFF(String outputPath) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(outputPath));
        
        // Write header
        bw.write("@relation happiness_data\n\n");
        
        // Detect attribute types
        for (int i = 0; i < headers.length; i++) {
            String attrType = detectColumnType(i);
            bw.write("@attribute " + sanitizeAttributeName(headers[i]) + " " + attrType + "\n");
        }
        
        bw.write("\n@data\n");
        
        // Write data
        for (String[] row : data) {
            List<String> values = new ArrayList<>();
            for (String value : row) {
                if (value == null || value.trim().isEmpty()) {
                    values.add("?");
                } else {
                    values.add(value);
                }
            }
            bw.write(String.join(",", values) + "\n");
        }
        
        bw.close();
        System.out.println("ARFF file created: " + outputPath);
    }
    
    private String detectColumnType(int columnIndex) {
        int numericCount = 0;
        int totalCount = 0;
        
        for (String[] row : data) {
            if (row[columnIndex] != null && !row[columnIndex].trim().isEmpty()) {
                totalCount++;
                try {
                    Double.parseDouble(row[columnIndex]);
                    numericCount++;
                } catch (NumberFormatException e) {
                    // String type
                }
            }
        }
        
        if (totalCount > 0 && numericCount == totalCount) {
            return "numeric";
        } else {
            return "string";
        }
    }
    
    private String sanitizeAttributeName(String name) {
        return name.replaceAll("[^a-zA-Z0-9_]", "_");
    }
}
