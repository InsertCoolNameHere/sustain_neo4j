/* ========================================================
 * GBoostIncremental.java -
 *      Defines a generalized gradient boost regression model that can be
 *      built and executed over a set of MongoDB documents.
 *
 * Author: Saptashwa Mitra
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 * ======================================================== */
package org.sustain.modeling;

import com.mongodb.spark.MongoSpark;
import com.mongodb.spark.config.ReadConfig;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.GBTRegressionModel;
import org.apache.spark.ml.regression.GBTRegressor;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.storage.StorageLevel;
import org.sustain.util.Constants;
import org.sustain.util.FancyLogger;
import scala.collection.JavaConverters;
import scala.collection.Seq;

import org.sustain.SparkManager;
import org.sustain.SparkTask;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Future;

/**
 * Provides an interface for building generalized Gradient Boost Regression
 * models on data pulled in using Mongo's Spark Connector.
 */
public class GBoostIncremental{

    private Dataset<Row> mongoCollection;
    // DATABASE PARAMETERS
    protected static final Logger log = LogManager.getLogger(GBoostIncremental.class);
    private String database, collection, mongoUri;
    private String[] features;
    private String label, gisJoin;

    // MODEL PARAMETERS
    // Loss function which GBT tries to minimize. (case-insensitive) Supported: "squared" (L2) and "absolute" (L1) (default = squared)
    private String lossType = null;
    // Max number of iterations
    private Integer maxIter = null;
    //Minimum information gain for a split to be considered at a tree node. default 0.0
    private Double minInfoGain = null;
    // Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Must be at least 1. (default = 1)
    private Integer minInstancesPerNode = null;
    //Minimum fraction of the weighted sample count that each child must have after split. Should be in the interval [0.0, 0.5). (default = 0.0)
    private Double minWeightFractionPerNode = null;
    //Fraction of the training data used for learning each decision tree, in range (0, 1]. (default = 1.0)
    private Double subsamplingRate = null;
    //Param for Step size (a.k.a. learning rate) in interval (0, 1] for shrinking the contribution of each estimator. (default = 0.1)
    private Double stepSize = null;
    // Number of features to consider for splits at each node. Supported: "auto", "all", "sqrt", "log2", "onethird".
    // If "auto" is set, this parameter is set based on numTrees: if numTrees == 1, set to "all"; if numTrees > 1 (forest) set to "onethird".
    private String featureSubsetStrategy = null; //auto/all/sqrt/log2/onethird
    //Criterion used for information gain calculation. Supported values: "variance".
    private String impurity = null;
    //maxDepth - Maximum depth of the tree. (e.g., depth 0 means 1 leaf node, depth 1 means 1 internal node + 2 leaf nodes). (suggested value: 4)
    private Integer maxDepth = null;
    //maxBins - Maximum number of bins used for splitting features. (suggested value: 100)
    private Integer maxBins = null;
    private Double trainSplit = 0.8d;
    public double target_rmse;

    private GBTRegressor trained_gb;

    private GBTRegressionModel trained_gbModel;
    String filename = "";

    String queryField = "gis_join";
    //String queryField = "countyName";

    double rmse = 0.0;
    private double r2 = 0.0;

    public GBTRegressor getTrained_gb() {
        return trained_gb;
    }

    public GBTRegressionModel getTrained_gbModel() {
        return trained_gbModel;
    }

    public Dataset<Row> getMongoCollection() {
        return mongoCollection;
    }

    public void setMongoCollection(Dataset<Row> mongoCollection) {
        this.mongoCollection = mongoCollection;
    }

    public String getLossType() {
        return lossType;
    }

    public void setLossType(String lossType) {
        this.lossType = lossType;
    }

    public Integer getMaxIter() {
        return maxIter;
    }

    public void setMaxIter(Integer maxIter) {
        this.maxIter = maxIter;
    }

    public Double getMinInfoGain() {
        return minInfoGain;
    }

    public void setMinInfoGain(Double minInfoGain) {
        this.minInfoGain = minInfoGain;
    }

    public Integer getMinInstancesPerNode() {
        return minInstancesPerNode;
    }

    public void setMinInstancesPerNode(Integer minInstancesPerNode) {
        this.minInstancesPerNode = minInstancesPerNode;
    }

    public Double getMinWeightFractionPerNode() {
        return minWeightFractionPerNode;
    }

    public void setMinWeightFractionPerNode(Double minWeightFractionPerNode) {
        this.minWeightFractionPerNode = minWeightFractionPerNode;
    }

    public double getR2() {
        return r2;
    }

    public void setR2(double r2) {
        this.r2 = r2;
    }

    public double getRmse() {
        return rmse;
    }

    public void setRmse(double rmse) {
        this.rmse = rmse;
    }

    public void setTrainSplit(Double trainSplit) {
        this.trainSplit = trainSplit;
    }

    public GBoostIncremental(String mongoUri, String database, String collection, String gisJoin) {
        log.info("Gradient Boosting constructor invoked");
        setMongoUri(mongoUri);
        setDatabase(database);
        setCollection(collection);
        setGisjoin(gisJoin);
    }

    public String getDatabase() {
        return database;
    }

    public void setDatabase(String database) {
        this.database = database;
    }

    public String getCollection() {
        return collection;
    }

    public void setCollection(String collection) {
        this.collection = collection;
    }

    public String getMongoUri() {
        return mongoUri;
    }

    public void setMongoUri(String mongoUri) {
        this.mongoUri = mongoUri;
    }

    public void setFeatures(String[] features) {
        this.features = features;
    }

    public void setGisjoin(String gisJoin) {
        this.gisJoin = gisJoin;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public String[] getFeatures() {
        return features;
    }

    public String getGisJoin() {
        return gisJoin;
    }

    public String getLabel() {
        return label;
    }

    public Double getSubsamplingRate() {
        return subsamplingRate;
    }

    public void setSubsamplingRate(Double subsamplingRate) {
        this.subsamplingRate = subsamplingRate;
    }

    public Double getStepSize() {
        return stepSize;
    }

    public void setStepSize(Double stepSize) {
        this.stepSize = stepSize;
    }

    public String getFeatureSubsetStrategy() {
        return featureSubsetStrategy;
    }

    public void setFeatureSubsetStrategy(String featureSubsetStrategy) {
        this.featureSubsetStrategy = featureSubsetStrategy;
    }

    public String getImpurity() {
        return impurity;
    }

    public void setImpurity(String impurity) {
        this.impurity = impurity;
    }

    public Integer getMaxDepth() {
        return maxDepth;
    }

    public void setMaxDepth(Integer maxDepth) {
        this.maxDepth = maxDepth;
    }

    public Integer getMaxBins() {
        return maxBins;
    }

    public void setMaxBins(Integer maxBins) {
        this.maxBins = maxBins;
    }

    private Seq<String> desiredColumns() {
        List<String> cols = new ArrayList<>();
        cols.add(queryField);
        Collections.addAll(cols, this.features);
        cols.add(this.label);
        return convertListToSeq(cols);
    }

    /**
     * Converts a Java List<String> of inputs to a Scala Seq<String>
     * @param inputList The Java List<String> we wish to transform
     * @return A Scala Seq<String> representing the original input list
     */
    public Seq<String> convertListToSeq(List<String> inputList) {
        return JavaConverters.asScalaIteratorConverter(inputList.iterator()).asScala().toSeq();
    }

    private String fancy_logging(String msg){

        String logStr = "\n============================================================================================================\n";
        logStr+=msg;
        logStr+="\n============================================================================================================";

        log.info(logStr);
        return logStr;
    }

    private double calc_interval(double startTime) {
        return ((double)System.currentTimeMillis() - startTime)/1000;
    }

    /**
     * Creates Spark context and trains the distributed model
     */

    public Boolean train() {
        if(trained_gbModel == null)
            return train_parent();
        else
            return train_child();

    }


    public Boolean train_parent() {
        String full_summary = "";
        //addClusterDependencyJars(sparkContext);
        double startTime = System.currentTimeMillis();

        full_summary +=fancy_logging("Initiating Exhaustive Gradient Boost Parent Modelling...");

        // Select just the columns we want, discard the rest
        Dataset<Row> selected = mongoCollection.select("_id", desiredColumns());

        full_summary +=fancy_logging("Data Fetch Completed in "+ calc_interval(startTime)+" secs");
        startTime = System.currentTimeMillis();

        Dataset<Row> gisDataset = selected.filter(selected.col(queryField).equalTo(gisJoin))
                .withColumnRenamed(this.label, "label"); // Rename the chosen label column to "label"

        log.info("DATA TYPES: \n"+Arrays.toString(gisDataset.columns())+" "+gisDataset.dtypes());

        // Create a VectorAssembler to assemble all the feature columns into a single column vector named "features"
        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(this.features)
                .setOutputCol("features");

        // Transform the gisDataset to have the new "features" column vector
        Dataset<Row> mergedDataset = vectorAssembler.transform(gisDataset);


        Dataset<Row>[] rds = mergedDataset.randomSplit(new double[]{trainSplit , 1.0d - trainSplit});
        Dataset<Row> trainrdd = rds[0].persist(StorageLevel.MEMORY_ONLY());
        Dataset<Row> testrdd = rds[1];

        fancy_logging("TRAINING SIZE:"+trainrdd.count());

        full_summary +=fancy_logging("Data Manipulation completed in "+calc_interval(startTime)+" secs"/*+"\nData Size: "+gisDataset.count()*/);
        startTime = System.currentTimeMillis();

        GBTRegressor gb = new GBTRegressor().setFeaturesCol("features").setLabelCol("label");

        // POPULATING USER PARAMETERS
        ingestParameters(gb);

        GBTRegressionModel gbModel = gb.fit(trainrdd);

        full_summary +=fancy_logging("Model Training completed in "+calc_interval(startTime));
        startTime = System.currentTimeMillis();

        Dataset<Row> pred_pair = gbModel.transform(testrdd).select("label", "prediction").cache();

        RegressionMetrics metrics = new RegressionMetrics(pred_pair);

        this.rmse = metrics.rootMeanSquaredError();
        this.r2 = metrics.r2();
        full_summary +=fancy_logging("Model Testing/Loss Computation completed in "+calc_interval(startTime)+"\nEVALUATIONS: RMSE, R2: "+rmse+" "+r2);

        logModelResults();
        this.trained_gb = gb;
        this.trained_gbModel = gbModel;

        System.out.println(full_summary);
        FancyLogger.write_out(full_summary+"\n", filename);
        return true;
    }

    public Boolean train_child() {
        //addClusterDependencyJars(sparkContext);
        double startTime = System.currentTimeMillis();

        String msg = "";
        msg = "Initiating Incremental Gradient Boost Child Modelling...";

        FancyLogger.write_out(fancy_logging(msg), filename);

        // Select just the columns we want, discard the rest
        Dataset<Row> selected = mongoCollection.select("_id", desiredColumns());

        msg = "Data Fetch Completed in "+ calc_interval(startTime)+" secs";

        FancyLogger.write_out(fancy_logging(msg), filename);
        startTime = System.currentTimeMillis();

        // Create a VectorAssembler to assemble all the feature columns into a single column vector named "features"
        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(this.features)
                .setOutputCol("features");

        msg = "Data Manipulation completed in "+calc_interval(startTime)+" secs"/*+"\nData Size: "+gisDataset.count()*/;

        FancyLogger.write_out(fancy_logging(msg), filename);

        /* ITERATIVE SAMPLING OF THE mergedDataset*/
        float trainFraction = 0.15f;
        // PREPARING DATASET2
        Dataset<Row> gisDataset2 = selected.filter(selected.col(queryField).equalTo(gisJoin))
                .withColumnRenamed(this.label, "label"); // Rename the chosen label column to "label"

        //gisDataset2 = gisDataset2.sample(0.1);

        Dataset<Row> mergedDataset_transfer = vectorAssembler.transform(gisDataset2).cache();

        // COPYING PARAMETERS FROM PRE_TRAINED MODEL
        GBTRegressionModel gb2_model = trained_gbModel.copy(new ParamMap());
        GBTRegressor gb2 = (GBTRegressor) gb2_model.parent();

        int iter = 0;
        double targetRMSE = this.rmse;
        String fullSummary = "";
        fullSummary += FancyLogger.fancy_logging("Target RMSE:  " + targetRMSE+"\n");

        long startTime_overall = System.currentTimeMillis();
        while (true) {
            startTime = System.currentTimeMillis();
            Dataset<Row> workingDataset = mergedDataset_transfer.sample(trainFraction);
            Dataset<Row>[] rds_transfer = workingDataset.randomSplit(new double[]{trainSplit, 1.0d - trainSplit});
            Dataset<Row> trainrdd_transfer = rds_transfer[0].cache();
            Dataset<Row> testrdd_transfer = rds_transfer[1];
            fancy_logging("Model Data Split completed in " + calc_interval(startTime));

            /* TRAIN PHASE */

            GBTRegressionModel gb2Model_iter = gb2.fit(trainrdd_transfer);
            fullSummary += FancyLogger.fancy_logging("Model Training Round " + iter + " completed in " + calc_interval(startTime)) +"\n";

            /* TEST/EVALUATION PHASE */
            startTime = System.currentTimeMillis();
            Dataset<Row> pred_pair = gb2Model_iter.transform(testrdd_transfer).select("label", "prediction").cache();
            RegressionMetrics metrics = new RegressionMetrics(pred_pair);
            this.rmse = metrics.rootMeanSquaredError();

            fullSummary += FancyLogger.fancy_logging("Model Evaluation/Loss Computation Round " + iter + " completed in " + calc_interval(startTime)
                    + "\nEVALUATIONS: RMSE, R2: " + rmse + " " + r2) + "\n";

            logModelResults();

            trainFraction = trainFraction * 2;

            if (this.rmse < targetRMSE*1.05) {
                fullSummary += FancyLogger.fancy_logging("DESIRED ACCURACY ACHIEVED... EVALUATION TIME"+targetRMSE+" "+this.rmse) + "\n";
                break;
            } else if (trainFraction > 0.9) {
                fullSummary += FancyLogger.fancy_logging("DESIRED ACCURACY NOT ACHIEVED... RAN OUT OF SAMPLES") + "\n";
                break;
            } else {
                fullSummary += FancyLogger.fancy_logging("DESIRED ACCURACY NOT ACHIEVED " + targetRMSE+" "+this.rmse + " ...RETRAINING") + "\n";

            }

            // COPYING OVER PARAMETERS FROM PREVIOUS ITERATION
            gb2 = (GBTRegressor) gb2Model_iter.parent();
            iter++;
        }

        fullSummary+= FancyLogger.fancy_logging("OVERALL CONVERGENCE TIME " + calc_interval(startTime_overall)
                + "\nEVALUATIONS: RMSE, R2: " + rmse + " " + r2) + "\n";
        FancyLogger.write_out(fullSummary,filename);


        return true;
    }

    private void addClusterDependencyJars(JavaSparkContext sparkContext) {
        String[] jarPaths = {
                "build/libs/mongo-spark-connector_2.12-3.0.1.jar",
                "build/libs/spark-core_2.12-3.0.1.jar",
                "build/libs/spark-mllib_2.12-3.0.1.jar",
                "build/libs/spark-sql_2.12-3.0.1.jar",
                "build/libs/bson-4.0.5.jar",
                "build/libs/mongo-java-driver-3.12.5.jar",
                //"build/libs/mongodb-driver-core-4.0.5.jar"
        };

        for (String jar: jarPaths) {
            log.info("Adding dependency JAR to the Spark Context: {}", jar);
            sparkContext.addJar(jar);
        }
    }

    /**
     * Injecting user-defined parameters into model
     * @param gb - Gradient Boosting Regression model Object
     */
    private void ingestParameters(GBTRegressor gb) {
        if (this.subsamplingRate != null) {
            gb.setSubsamplingRate(this.subsamplingRate);
        }
        if (this.stepSize != null) {
            gb.setStepSize(this.stepSize);
        }
        if (this.featureSubsetStrategy != null) {
            gb.setFeatureSubsetStrategy(this.featureSubsetStrategy);
        }
        if (this.impurity != null) {
            gb.setImpurity(this.impurity);
        }
        if (this.maxDepth != null) {
            gb.setMaxDepth(this.maxDepth);
        }
        if (this.maxBins != null) {
            gb.setMaxBins(this.maxBins);
        }

        if (this.minInfoGain != null) {
            gb.setMinInfoGain(this.minInfoGain);
        }

        if (this.minInstancesPerNode != null) {
            gb.setMinInstancesPerNode(this.minInstancesPerNode);
        }

        if (this.minWeightFractionPerNode != null) {
            gb.setMinWeightFractionPerNode(this.minWeightFractionPerNode);
        }

        if (this.lossType != null) {
            gb.setLossType(this.lossType);
        }
        if (this.maxIter != null) {
            gb.setMaxIter(this.maxIter);
        }

    }

    public void populateTest() {
        this.maxIter = 5;
    }

    private void logModelResults() {
        log.info("Results for GISJoin {}\n" +
                        "RMSE: {}\n" +
                        "R2: {}\n"
                ,
                this.gisJoin, this.rmse, this.r2);
    }

    /**
     * Used exclusively for testing and running a linear model directly, without having to interface with gRPC.
     * @param args Usually not used.
     */
    public static void main(String[] args) {
        FancyLogger.base_path = "/s/parsons/b/others/sustain/sapmitra/";
        String[] features = {"relative_humidity_percent","orography_surface_level_meters","10_metre_u_wind_component_meters_per_second","10_metre_v_wind_component_meters_per_second","pressure_pascal","visibility_meters","total_cloud_cover_percent"};
        String label = "temp_surface_level_kelvin";
        String gisJoins = "G3000710";
        String collection_name = "noaa_nam";

        SparkSession sparkSession = SparkSession.builder()
                .master("local[*]")
                .appName("SUSTAIN Local GBoost")
                .config("spark.executor.cores", "8")
                .config("spark.executor.memory", "16G")
                .config("spark.mongodb.input.uri", "mongodb://lattice-100:27018/") // default mongos router
                .config("spark.mongodb.input.database", "sustaindb") // sustaindb
                .config("spark.mongodb.input.collection", "noaa_nam")
                .getOrCreate();

        JavaSparkContext sparkContext = new JavaSparkContext(sparkSession.sparkContext());
        ReadConfig readConfig = ReadConfig.create(sparkContext);

        GBoostIncremental gbModel = new GBoostIncremental(
                "mongodb://lattice-46:27017", "sustaindb", collection_name, gisJoins);

        Dataset<Row> mongoCollection = MongoSpark.load(sparkContext, readConfig).toDF();
        gbModel.setMongoCollection(mongoCollection);

        gbModel.filename = "local/"+gisJoins+".txt";


        gisJoins = "G4601370";
        //gbModel.populateTest();
        gbModel.setFeatures(features);
        gbModel.setLabel(label);
        gbModel.setGisjoin(gisJoins);

        gbModel.train();
        log.info("Executed rfModel.main() successfully");
        sparkContext.close();

        mongoCollection.unpersist();
        try {
            Thread.sleep(10000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        JavaSparkContext sparkContext2 = new JavaSparkContext(sparkSession.sparkContext());
        ReadConfig readConfig2 = ReadConfig.create(sparkContext);

        GBoostIncremental gbModel2 = new GBoostIncremental(
                "mongodb://lattice-46:27017", "sustaindb", collection_name, gisJoins);
        gbModel2.setMongoCollection(MongoSpark.load(sparkContext2, readConfig2).toDF());

        gbModel2.filename = "local/"+gisJoins+".txt";

        //gbModel.populateTest();
        gbModel2.setFeatures(features);
        gbModel2.setLabel(label);
        gbModel2.setGisjoin(gisJoins);
        gbModel2.trained_gbModel = gbModel.trained_gbModel;
        gbModel2.target_rmse = gbModel.rmse;

        gbModel2.train();
        log.info("Executed rfModel.main() successfully");
        sparkContext2.close();
    }

}