package sustain.census;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.sustain.census.CensusGrpc;
import org.sustain.census.CensusResolution;
import org.sustain.census.CensusServer;
import org.sustain.census.ClientHelper;
import org.sustain.census.Constants;
import org.sustain.census.Decade;
import org.sustain.census.Predicate;
import org.sustain.census.TargetedQueryResponse;
import org.sustain.census.db.Util;

import java.util.List;

import static sustain.census.TestUtil.decades;

public class TargetedPopulationQueryTests {
    private static final Logger log = LogManager.getLogger(TargetedPopulationQueryTests.class);

    private static CensusGrpc.CensusBlockingStub censusBlockingStub;
    private static ManagedChannel channel;
    private static CensusServer server;
    private static ClientHelper clientHelper;

    @BeforeAll
    static void init() throws InterruptedException {
        server = new CensusServer();
        new ServerRunner(server).start();
        Thread.sleep(2000);
        String target = Util.getProperty(Constants.Server.HOST) + ":" + Constants.Server.PORT;
        channel = ManagedChannelBuilder.forTarget(target).usePlaintext().build();
        censusBlockingStub = CensusGrpc.newBlockingStub(channel);
        clientHelper = new ClientHelper(censusBlockingStub);
    }

    @AfterAll
    static void cleanUp() {
        server.shutdownNow();
    }

    @Test
    public void testStatePopulationTargeted() {
        for (Decade decade : decades) {
            TargetedQueryResponse targetedQueryResponse = clientHelper.requestTargetedInfo(
                    Predicate.Feature.Population,
                    CensusResolution.State,
                    decade,
                    Predicate.ComparisonOperator.GREATER_THAN,
                    10000000);
            log.info("Tests: States where population is greater than 10 million in " + decade.toString());
            Assertions.assertNotNull(targetedQueryResponse);

            List<TargetedQueryResponse.SpatialInfo> spatialInfoList = targetedQueryResponse.getSpatialInfoList();

            for (TargetedQueryResponse.SpatialInfo spatialInfo : spatialInfoList) {
                Assertions.assertNotNull(spatialInfo);
                Assertions.assertNotEquals("", spatialInfo.getGeoId());
                Assertions.assertNotEquals("", spatialInfo.getName());
            }
        }
    }

    @Test
    public void testCountyPopulationTargeted() {
        for (Decade decade : decades) {
            TargetedQueryResponse targetedQueryResponse = clientHelper.requestTargetedInfo(
                    Predicate.Feature.Population,
                    CensusResolution.County,
                    decade,
                    Predicate.ComparisonOperator.GREATER_THAN,
                    10000000);
            log.info("Tests: States where population is greater than 10 million in " + decade.toString());
            Assertions.assertNotNull(targetedQueryResponse);

            List<TargetedQueryResponse.SpatialInfo> spatialInfoList = targetedQueryResponse.getSpatialInfoList();

            for (TargetedQueryResponse.SpatialInfo spatialInfo : spatialInfoList) {
                Assertions.assertNotNull(spatialInfo);
                Assertions.assertNotEquals("", spatialInfo.getGeoId());
                Assertions.assertNotEquals("", spatialInfo.getName());
            }
        }
    }


    @Test
    public void testTractPopulationTargeted() {
        for (Decade decade : decades) {
            TargetedQueryResponse targetedQueryResponse = clientHelper.requestTargetedInfo(
                    Predicate.Feature.Population,
                    CensusResolution.Tract,
                    decade,
                    Predicate.ComparisonOperator.GREATER_THAN,
                    10000000);
            log.info("Tests: States where population is greater than 10 million in " + decade.toString());
            Assertions.assertNotNull(targetedQueryResponse);

            List<TargetedQueryResponse.SpatialInfo> spatialInfoList = targetedQueryResponse.getSpatialInfoList();

            for (TargetedQueryResponse.SpatialInfo spatialInfo : spatialInfoList) {
                Assertions.assertNotNull(spatialInfo);
                Assertions.assertNotEquals("", spatialInfo.getGeoId());
                Assertions.assertNotEquals("", spatialInfo.getName());
            }
        }
    }
}
