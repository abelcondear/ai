package model;

import org.pmml4s.model.Model;
import java.util.Arrays;
import java.util.Map;

public class ModelUtils {
    public ModelUtils() {
        //TODO
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

