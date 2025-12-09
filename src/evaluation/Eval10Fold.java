package evaluation;

import java.util.*;
import java.util.stream.*;

import Model.PCA;
import Model.PCA_RandomForest;
import Model.RandomForestClassifier;

public class Eval10Fold {

    // ==== LOAD DATA DÙNG CHUNG ====
    public static void loadData(
            String filePath,
            List<double[]> X,
            List<String> y) throws Exception {

        PCA_RandomForest.loadData(filePath, X, y);
    }  
 // ==== 10-FOLD CROSS VALIDATION: PCA + RANDOM FOREST (FULL METRICS) ====
    public static void evaluatePCA_RandomForest(
            List<double[]> X,
            List<String> y,
            int nComponents,
            int numTrees,
            int maxDepth,
            int minSamplesSplit) {

        System.out.println("\n==============================");
        System.out.println(" 10-FOLD CV: PCA + RANDOM FOREST");
        System.out.println("==============================");

        int n = X.size();
        int foldSize = n / 10;

        List<Integer> idx = IntStream.range(0, n)
                .boxed()
                .collect(Collectors.toList());
        Collections.shuffle(idx, new Random(42));

        double totalAcc = 0;
        double totalF1 = 0;
        long totalTrainTime = 0;
        long totalPredictTime = 0;

        for (int fold = 0; fold < 10; fold++) {

            int start = fold * foldSize;
            int end = (fold == 9) ? n : start + foldSize;

            Set<Integer> testSet = new HashSet<>(idx.subList(start, end));

            List<double[]> X_train = new ArrayList<>();
            List<String> y_train = new ArrayList<>();
            List<double[]> X_test = new ArrayList<>();
            List<String> y_test = new ArrayList<>();

            for (int i = 0; i < n; i++) {
                if (testSet.contains(i)) {
                    X_test.add(X.get(i));
                    y_test.add(y.get(i));
                } else {
                    X_train.add(X.get(i));
                    y_train.add(y.get(i));
                }
            }

            PCA pca = new PCA(nComponents);
            pca.fit(X_train);

            List<double[]> X_train_pca = pca.transform(X_train);
            List<double[]> X_test_pca = pca.transform(X_test);

            int maxFeatures = (int) Math.max(1, Math.sqrt(nComponents));

            RandomForestClassifier model = new RandomForestClassifier(
                    numTrees, maxDepth, minSamplesSplit, maxFeatures);

            long t1 = System.currentTimeMillis();
            model.fit(X_train_pca, y_train);
            long trainTime = System.currentTimeMillis() - t1;
            totalTrainTime += trainTime;

            long t2 = System.currentTimeMillis();
            List<String> y_pred = model.predict(X_test_pca);
            long predictTime = System.currentTimeMillis() - t2;
            totalPredictTime += predictTime;

            double acc = model.calculateAccuracy(y_test, y_pred);
            totalAcc += acc;

            List<String> unique = y_train.stream()
                    .distinct()
                    .collect(Collectors.toList());

            Map<String, Map<String, Double>> report =
                    model.calculateClassificationReport(y_test, y_pred);

            double f1 = report.get("macro_avg").get("f1");
            totalF1 += f1;
            System.out.printf("Fold %d | Acc: %.4f | F1: %.4f | Train: %d ms | Predict: %d ms\n",
                    fold + 1, acc, f1, trainTime, predictTime);
        }

        System.out.println("---------------------------------");
        System.out.printf("Average PCA+RF Accuracy: %.4f\n", totalAcc / 10);
        System.out.printf("Average PCA+RF F1-score: %.4f\n", totalF1 / 10);
        System.out.printf("Total Train Time: %d ms\n", totalTrainTime);
        System.out.printf("Total Predict Time:%d ms\n", totalPredictTime);
        System.out.println("=================================\n");
    }
    // ==== HÀM TỔNG — MAIN CHỈ GỌI HÀM NÀY ====
    public static void runEvaluation(String cleanedCSV) throws Exception {

        List<double[]> X = new ArrayList<>();
        List<String> y = new ArrayList<>();
        loadData(cleanedCSV, X, y);

        evaluatePCA_RandomForest(
                X, y,
                5,      // nComponents
                50,     // numTrees
                10,     // maxDepth
                2       // minSamplesSplit
        );

    }
}
