package org.sustain.handlers;

import com.mongodb.spark.MongoSpark;
import com.mongodb.spark.config.ReadConfig;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.sustain.Collection;
import org.sustain.LinearRegressionRequest;
import org.sustain.LinearRegressionResponse;
import org.sustain.ModelRequest;
import org.sustain.ModelResponse;
import org.sustain.ModelType;
import org.sustain.modeling.LinearRegressionModelImpl;
import org.sustain.util.Constants;
import org.sustain.util.Profiler;

import java.util.HashMap;
import java.util.Map;

public class RegressionQueryHandler extends ModelHandler {

    private static final Logger log = LogManager.getLogger(RegressionQueryHandler.class);

    public RegressionQueryHandler(ModelRequest request, StreamObserver<ModelResponse> responseObserver,
                                  JavaSparkContext sparkContext) {
        super(request, responseObserver, sparkContext);
    }

    @Override
    public void handleRequest() {
        if (isValid(this.request)) {
            logRequest(this.request);
            Profiler profiler = new Profiler();

            // Set parameters of Linear Regression Model
            LinearRegressionRequest lrRequest = this.request.getLinearRegressionRequest();
            Collection requestCollection = this.request.getCollections(0); // We only support 1 collection currently

            String mongoUri = String.format("mongodb://%s:%s", Constants.DB.HOST, Constants.DB.PORT);

            // Create a custom ReadConfig
            profiler.addTask("Create ReadConfig");
            Map<String, String> readOverrides = new HashMap<String, String>();
            readOverrides.put("uri", mongoUri);
            readOverrides.put("database", Constants.DB.NAME);
            readOverrides.put("collection", requestCollection.getName());
            ReadConfig readConfig = ReadConfig.create(this.sparkContext.getConf(), readOverrides);
            profiler.completeTask("Create ReadConfig");

            // Lazy-load the collection in as a DF
            profiler.addTask("Load mongoCollection");
            Dataset<Row> mongoCollection = MongoSpark.load(sparkContext, readConfig).toDF();
            profiler.completeTask("Load mongoCollection");

            // Build and run a model for each GISJoin in the request

            for (String gisJoin: lrRequest.getGisJoinsList()) {

                LinearRegressionModelImpl model = new LinearRegressionModelImpl.LinearRegressionModelBuilder()
                        .forMongoCollection(mongoCollection)
                        .forGISJoin(gisJoin)
                        .forFeatures(requestCollection.getFeaturesList())
                        .forLabel(requestCollection.getLabel())
                        .withLoss(lrRequest.getLoss())
                        .withSolver(lrRequest.getSolver())
                        .withAggregationDepth(lrRequest.getAggregationDepth())
                        .withMaxIterations(lrRequest.getMaxIterations())
                        .withElasticNetParam(lrRequest.getElasticNetParam())
                        .withEpsilon(lrRequest.getEpsilon())
                        .withRegularizationParam(lrRequest.getRegularizationParam())
                        .withTolerance(lrRequest.getConvergenceTolerance())
                        .withFitIntercept(lrRequest.getFitIntercept())
                        .withStandardization(lrRequest.getSetStandardization())
                        .build();

                String buildAndRunModelTaskName = String.format("buildAndRunModel [%s]", gisJoin);
                profiler.addTask(buildAndRunModelTaskName);
                model.buildAndRunModel(); // Launches the Spark Model
                profiler.completeTask(buildAndRunModelTaskName);

                LinearRegressionResponse modelResults = LinearRegressionResponse.newBuilder()
                        .setGisJoin(model.getGisJoin())
                        .setTotalIterations(model.getTotalIterations())
                        .setRmseResidual(model.getRmse())
                        .setR2Residual(model.getR2())
                        .setIntercept(model.getIntercept())
                        .addAllSlopeCoefficients(model.getCoefficients())
                        .addAllObjectiveHistory(model.getObjectiveHistory())
                        .build();

                ModelResponse response = ModelResponse.newBuilder()
                        .setLinearRegressionResponse(modelResults)
                        .build();

                logResponse(response);
                this.responseObserver.onNext(response);
            }
        } else {
            log.warn("Invalid Model Request!");
        }
    }


    @Override
    boolean isValid(ModelRequest modelRequest) {
        if (modelRequest.getType().equals(ModelType.LINEAR_REGRESSION)) {
            if (modelRequest.getCollectionsCount() == 1) {
                if (modelRequest.getCollections(0).getFeaturesCount() == 1) {
                    return modelRequest.hasLinearRegressionRequest();
                }
            }
        }
        return false;
    }
}
