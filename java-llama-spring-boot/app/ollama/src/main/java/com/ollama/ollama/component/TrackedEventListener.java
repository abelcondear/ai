package com.ollama.ollama.component;

import com.ollama.ollama.configuration.TaskStatus;
import com.ollama.ollama.utils.TrackedEvent;
import com.ollama.ollama.utils.OllamaReader;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;
import java.io.IOException;

@Component
public class TrackedEventListener {

    private final TaskStatusStore statusStore;
    private final TaskPromptStore promptStore;

    public TrackedEventListener(TaskStatusStore statusStore, TaskPromptStore promptStore) {
        this.statusStore = statusStore;
        this.promptStore = promptStore;
    }

    @Async
    @EventListener
    public void handleTrackedEvent(TrackedEvent event) throws IOException {
        String taskId = event.getTaskId();

        try {
            statusStore.setStatus(taskId, TaskStatus.RUNNING);

            System.out.println("Processing task " + taskId);
            // set working process from llama source

            OllamaReader ollamaReader = new OllamaReader(event.getTaskPrompt().getPrompt());

            // need to create a thread to run async the ollama execution
            // on the other way, it pauses until the ollama execution is finished

            Thread.sleep(100); // Simulate long task

            event.getTaskPrompt().setResponse(ollamaReader.response);
            promptStore.setPrompt(taskId, event.getTaskPrompt());
            statusStore.setStatus(taskId, TaskStatus.COMPLETED);
            System.out.println("Task " + taskId + " completed.");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            statusStore.setStatus(taskId, TaskStatus.FAILED);
            System.err.println("Task " + taskId + " failed: " + e.getMessage());
        }
    }
}