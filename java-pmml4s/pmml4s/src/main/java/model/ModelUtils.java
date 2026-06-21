// ----
package model;
// ----
import org.pmml4s.model.Model;
// ----
import java.io.File;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.Map;
// ----
import file.FileUtils;
// ----

public class ModelUtils {
    public ModelUtils() { /* TODO */ }

    public Model createModel(String filename) throws URISyntaxException {
        FileUtils fileUtils = new FileUtils();

        return Model.fromFile(
                new File(
                        fileUtils.GetFilePath(filename)
                )
        );
    }

    public Map<String, Double> createValues()
    {
        Map<String, Double> values = Map.of
                (
                        "buoy_day_ID", 776d,
                        "buoy", 59d,
                        "day", 7d,
                        "latitude", -8.03d,
                        "longitude", 164.82d,
                        "zon_winds", 0d,
                        "mer_winds", 0d,
                        "humidity", 89.50d,
                        "airtemp", 27.51d,
                        "s_s_temp",  0d // this value will be predicted
                );

        return values;
    }

    public Double getRegressionValue
            (
                    Model model,
                    Map<String, Double> values
            ) {
        Object[] valuesMap =
                Arrays.stream
                (
                    model.inputNames()
                )
                .map(values::get)
                .toArray();

        Object[] result = model.predict(valuesMap);

        return (Double) result[0];
    }
}

