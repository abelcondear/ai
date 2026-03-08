package com.ollama.ollama.error;

public class ShellExecutionException extends RuntimeException {
    public ShellExecutionException(String message) {
        super(String.format("%s", message));
    }
}