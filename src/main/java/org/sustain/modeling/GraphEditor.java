/* ========================================================
 * GBoostRegressionModel.java -
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
import com.mongodb.spark.config.WriteConfig;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.GBTRegressionModel;
import org.apache.spark.ml.regression.GBTRegressor;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.storage.StorageLevel;
import org.sustain.util.Constants;
import org.sustain.util.FancyLogger;
import scala.collection.JavaConverters;
import scala.collection.Seq;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

// PERFORMS EXHAUSTIVE TRAINING OVER A SET OF PARAMETERS
/**
 * Provides an interface for building generalized Gradient Boost Regression
 * models on data pulled in using Mongo's Spark Connector.
 */
public class GraphEditor {

    private String filename="";

    private Dataset<Row> mongoCollection_points;
    private Dataset<Row> mongoCollection_lines;

    // DATABASE PARAMETERS
    protected static final Logger log = LogManager.getLogger(GraphEditor.class);
    private String database, mongoUri;
    private String[] features;
    private String label, gisJoin;
    WriteConfig wc;

    public WriteConfig getWc() {
        return wc;
    }

    public void setWc(WriteConfig wc) {
        this.wc = wc;
    }

    public Dataset<Row> getMongoCollection_points() {
        return mongoCollection_points;
    }

    public void setMongoCollection_points(Dataset<Row> mongoCollection_points) {
        this.mongoCollection_points = mongoCollection_points;
    }

    public Dataset<Row> getMongoCollection_lines() {
        return mongoCollection_lines;
    }

    public void setMongoCollection_lines(Dataset<Row> mongoCollection_lines) {
        this.mongoCollection_lines = mongoCollection_lines;
    }

    public GraphEditor() {
        log.info("Gradient Boosting constructor invoked");
        filename = "inside_editor.txt";
    }

    public String getDatabase() {
        return database;
    }

    public void setDatabase(String database) {
        this.database = database;
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
    public Boolean manipulate() {
        //addClusterDependencyJars(sparkContext);
        double startTime = System.currentTimeMillis();

        String msg = "";
        msg = "Initiating Graph Manipulation...";
        fancy_logging(msg);
        FancyLogger.write_out(msg, filename);

        // Select just the columns we want, discard the rest
        Dataset<Row> points = mongoCollection_points;
        Dataset<Row> lines = mongoCollection_lines;

        Dataset<Row> filtered_points = points.select("_id", "geometry.coordinates", "properties")
                .withColumnRenamed("properties", "point_properties");

        FancyLogger.fancy_logging("POINT DATA: ", log);
        filtered_points.show(5);


        Dataset<Row> filtered_lines = lines.select("_id", "geometry.coordinates", "properties").withColumnRenamed("_id", "line_id");

        //filtered_lines.show(5);

        Dataset<Row> exploded_lines = filtered_lines.select(filtered_lines.col("line_id"), filtered_lines.col("properties"),
                org.apache.spark.sql.functions.explode(filtered_lines.col("coordinates")).as("coordinates"));

        FancyLogger.fancy_logging("EXPLODED DATA: ", log);
        exploded_lines.show(5);


        //Dataset<Row> joined_data = exploded_lines.join(filtered_points, "coordinates");

        Dataset<Row> joined_data = filtered_points.join(exploded_lines,
                filtered_points.col("coordinates").equalTo(exploded_lines.col("coordinates")), "leftouter");

        FancyLogger.fancy_logging("JOINED DATA: ", log);
        joined_data.show(5);

        //MongoSpark.save(joined_data, wc);

        return true;
    }

}