package com.bomdemo.kimiapi.model;

import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
public class Response {

    private String id;

    private String object;

    private int created;

    private String model;

    private List<ResponseChoice> choices;

    private UsageInfo usage;
}
