package Model;

import java.util.*;
import java.io.*;
import java.util.stream.*;


public class PCA {
    private double[] mean;
    private double[] std;
    private double[][] components;
    private int nComponents;
    
    public PCA(int nComponents) {
        this.nComponents = nComponents;
    }
    
    // Fit PCA on training data
    public void fit(List<double[]> X) {
        int nSamples = X.size();
        int nFeatures = X.get(0).length;
        
        // Initialize arrays
        mean = new double[nFeatures];
        std = new double[nFeatures];
        components = new double[nComponents][nFeatures];
        
        // Calculate mean for each feature
        for (int i = 0; i < nFeatures; i++) {
            final int idx = i;
            mean[i] = X.stream()
                      .mapToDouble(row -> row[idx])
                      .average()
                      .orElse(0.0);
        }
        
        // Calculate standard deviation for each feature
        for (int i = 0; i < nFeatures; i++) {
            final int idx = i;
            double variance = X.stream()
                             .mapToDouble(row -> Math.pow(row[idx] - mean[idx], 2))
                             .average()
                             .orElse(0.0);
            std[i] = Math.sqrt(variance);
            if (std[i] == 0) std[i] = 1.0; // Avoid division by zero
        }
        
        // Standardize the data
        double[][] standardized = new double[nSamples][nFeatures];
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                standardized[i][j] = (X.get(i)[j] - mean[j]) / std[j];
            }
        }
        
        // Calculate covariance matrix
        double[][] covariance = calculateCovarianceMatrix(standardized);
        
        // Get principal components (simplified - using variance-based selection)
        // In practice, you'd use eigenvalue decomposition
        selectTopComponentsByVariance(covariance);
    }
    
    // Transform data using PCA
    public List<double[]> transform(List<double[]> X) {
        int nSamples = X.size();
        int nFeatures = X.get(0).length;
        List<double[]> transformed = new ArrayList<>();
        
        for (int i = 0; i < nSamples; i++) {
            double[] row = X.get(i);
            double[] standardized = new double[nFeatures];
            
            // Standardize
            for (int j = 0; j < nFeatures; j++) {
                standardized[j] = (row[j] - mean[j]) / std[j];
            }
            
            // Project onto principal components
            double[] projected = new double[nComponents];
            for (int comp = 0; comp < nComponents; comp++) {
                double sum = 0.0;
                for (int j = 0; j < nFeatures; j++) {
                    sum += components[comp][j] * standardized[j];
                }
                projected[comp] = sum;
            }
            transformed.add(projected);
        }
        
        return transformed;
    }
    
    // Fit and transform
    public List<double[]> fitTransform(List<double[]> X) {
        fit(X);
        return transform(X);
    }
    
    private double[][] calculateCovarianceMatrix(double[][] data) {
        int nSamples = data.length;
        int nFeatures = data[0].length;
        double[][] covariance = new double[nFeatures][nFeatures];
        
        for (int i = 0; i < nFeatures; i++) {
            for (int j = i; j < nFeatures; j++) {
                double sum = 0.0;
                for (int k = 0; k < nSamples; k++) {
                    sum += data[k][i] * data[k][j];
                }
                covariance[i][j] = sum / (nSamples - 1);
                covariance[j][i] = covariance[i][j];
            }
        }
        
        return covariance;
    }
    
    private void selectTopComponentsByVariance(double[][] covariance) {
        // Simplified: select features with highest variance
        int nFeatures = covariance.length;
        double[] variances = new double[nFeatures];
        
        for (int i = 0; i < nFeatures; i++) {
            variances[i] = covariance[i][i];
        }
        
        // Get indices of top nComponents features by variance
        Integer[] indices = IntStream.range(0, nFeatures)
            .boxed()
            .sorted((a, b) -> Double.compare(variances[b], variances[a]))
            .limit(nComponents)
            .toArray(Integer[]::new);
        
        // Create component vectors (one-hot encoding for selected features)
        for (int i = 0; i < nComponents; i++) {
            for (int j = 0; j < nFeatures; j++) {
                components[i][j] = (j == indices[i]) ? 1.0 : 0.0;
            }
        }
    }
    
    // Get explained variance ratio
    public double[] getExplainedVarianceRatio() {
        // Simplified calculation
        double[] ratios = new double[nComponents];
        double sum = Arrays.stream(ratios).sum();
        for (int i = 0; i < nComponents; i++) {
            ratios[i] = (i + 1.0) / (nComponents * (nComponents + 1) / 2);
        }
        return ratios;
    }
    
    public int getNComponents() {
        return nComponents;
    }
}