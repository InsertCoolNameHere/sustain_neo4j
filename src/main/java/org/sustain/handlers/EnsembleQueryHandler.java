/* ========================================================
 * EnsembleQueryHandler.java
 *   Captures input parameters into out regression model object
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
package org.sustain.handlers;

import com.mongodb.spark.MongoSpark;
import com.mongodb.spark.config.ReadConfig;
import com.mongodb.spark.config.WriteConfig;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.sustain.*;
import org.sustain.modeling.*;
import org.sustain.util.Constants;
import org.sustain.util.CountyClusters;
import org.sustain.util.FancyLogger;

import java.util.*;
import java.util.concurrent.Future;

public class EnsembleQueryHandler extends GrpcSparkHandler<ModelRequest, ModelResponse> {

    private static final Logger log = LogManager.getLogger(EnsembleQueryHandler.class);
    private String filename = "neo_logs.txt";

    public static void main(String arg[]) {

        Map<String, List> opMap = CountyClusters.extractCountyGroups("./src/main/java/org/sustain/handlers/clusters.csv");
        System.out.println(opMap);
    }

    public EnsembleQueryHandler(ModelRequest request, StreamObserver<ModelResponse> responseObserver, SparkManager sparkManager) {
        super(request, responseObserver, sparkManager);
    }

    protected class GraphTask implements SparkTask<List<ModelResponse>> {
        @Override
        public List<ModelResponse> execute(JavaSparkContext sparkContext) throws Exception {

            String mongoUri = String.format("mongodb://%s:%d", Constants.DB.HOST, Constants.DB.PORT);
            String dbName = Constants.DB.NAME;

            // Initailize ReadConfig
            Map<String, String> readOverrides = new HashMap<String, String>();
            readOverrides.put("spark.mongodb.input.collection", "osm_points_geo");
            readOverrides.put("spark.mongodb.input.database", Constants.DB.NAME);
            readOverrides.put("spark.mongodb.input.uri", mongoUri);

            ReadConfig readConfig = ReadConfig.create(sparkContext.getConf(), readOverrides);

            // FETCHING MONGO COLLECTION ONCE FOR ALL MODELS
            Dataset<Row> point_collection = MongoSpark.load(sparkContext, readConfig).toDF();

            // Initailize ReadConfig
            Map<String, String> readOverrides_1 = new HashMap<String, String>();
            readOverrides_1.put("spark.mongodb.input.collection", "osm_lines_geo");
            readOverrides_1.put("spark.mongodb.input.database", Constants.DB.NAME);
            readOverrides_1.put("spark.mongodb.input.uri", mongoUri);

            ReadConfig readConfig_1 = ReadConfig.create(sparkContext.getConf(), readOverrides_1);


            Map<String, String> writeOverrides = new HashMap<String, String>();
            writeOverrides.put("spark.mongodb.output.collection", "demo_lines_inclusive");
            writeOverrides.put("spark.mongodb.output.database", Constants.DB.NAME);
            writeOverrides.put("spark.mongodb.output.uri", mongoUri);
            WriteConfig wc = WriteConfig.create(sparkContext.getConf(), writeOverrides);

            // FETCHING MONGO COLLECTION ONCE FOR ALL MODELS
            Dataset<Row> line_collection = MongoSpark.load(sparkContext, readConfig_1).toDF();

            List<ModelResponse> modelResponses = new ArrayList<>();

            GraphEditor ge = new GraphEditor();
            ge.setWc(wc);

            ge.setMongoCollection_points(point_collection.limit(10));
            ge.setMongoCollection_lines(line_collection.limit(10));

            /*FancyLogger.fancy_logging("POINT DATA: ", log);
            point_collection.show(5);

            FancyLogger.fancy_logging("LINE DATA: ", log);
            line_collection.show(5);*/

            // Submit task to Spark Manager
            boolean ok = ge.manipulate();

            if (ok) {

                FancyLogger.fancy_logging("WE ARE DONE !!!!!!");

            } else {
                log.info("Ran into a problem building a model for GISJoin {}, skipping.");
            }


            return modelResponses;
        }
    }

    /**
     * Checks the validity of a ModelRequest object, in the context of a Random Forest Regression request.
     * @param modelRequest The ModelRequest object populated by the gRPC endpoint.
     * @return Boolean true if the model request is valid, false otherwise.
     */
    @Override
    public boolean isValid(ModelRequest modelRequest) {
        if (modelRequest.getType().equals(ModelType.R_FOREST_REGRESSION) || modelRequest.getType().equals(ModelType.G_BOOST_REGRESSION)) {
            if (modelRequest.getCollectionsCount() == 1) {
                if (modelRequest.getCollections(0).getFeaturesCount() > 0) {
                    return (modelRequest.hasRForestRegressionRequest() || modelRequest.hasGBoostRegressionRequest());
                }
            }
        }

        return false;
    }


    @Override
    public void handleRequest() {

        String full_log_string = "";
        if (isValid(this.request)) {
            if(request.getType().equals(ModelType.G_BOOST_REGRESSION)) {

                try {
                    // ****************START PARENT TRAINING ***********************
                    List<Future<List<ModelResponse>>> batchedModelTasks_parents = new ArrayList<>();

                    GraphTask gTask = new GraphTask();
                    batchedModelTasks_parents.add(this.sparkManager.submit(gTask, "graph-join-query"));

                    // Wait for each task to complete and return their ModelResponses
                    for (Future<List<ModelResponse>> indvTask: batchedModelTasks_parents) {
                        List<ModelResponse> batchedModelResponses = indvTask.get();
                        for (ModelResponse modelResponse: batchedModelResponses) {
                            //DON'T THINK WE NEED RESPONSE OBSERVER HERE....NOTHING TO PASS BACK. JUST OBSERVE THE RESULTS
                            full_log_string+=FancyLogger.fancy_logging("RECEIVED A RESPONSE FOR "+modelResponse.getGBoostRegressionResponse().getGisJoin(),log);
                            this.responseObserver.onNext(modelResponse);
                        }
                    }

                    full_log_string+=FancyLogger.fancy_logging("GRAPH JOB CONCLUDED!!!!!", null);
                    System.out.println("GRAPH JOB CONCLUDED!!!!!");

                    FancyLogger.write_out(full_log_string, filename);

                } catch (Exception e) {
                    log.error("Failed to evaluate query", e);
                    responseObserver.onError(e);
                }
            }
        } else {
            log.warn("Invalid Model Request!");
        }
    }
}
