package com.ollama.ollama.controller;

import com.ollama.ollama.model.Ollama;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;
import java.io.IOException;

@RestController
public class OllamaController {
    @GetMapping("/ollama/{prompt}")
    public Ollama getResponse(@PathVariable("prompt") String prompt) throws IOException {
        return new Ollama(prompt);
    }
}
