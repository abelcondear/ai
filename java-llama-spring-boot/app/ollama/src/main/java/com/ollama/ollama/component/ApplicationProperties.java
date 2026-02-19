package com.ollama.ollama.component;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class ApplicationProperties {
    public static String AppName;

    @Value("${spring.application.name}")
    public void setAppName(String name) {
        AppName = name;
    }
}
