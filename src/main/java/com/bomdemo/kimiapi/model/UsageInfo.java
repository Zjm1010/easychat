package com.bomdemo.kimiapi.model;

import lombok.Getter;
import lombok.Setter;
import org.springframework.stereotype.Service;

@Getter
@Setter
public class UsageInfo {

    private int prompt_tokens;

    private int completion_tokens;

    private int total_tokens;
}
