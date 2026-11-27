#pragma once
struct Job {
    const char *expert;
    int begin, count;
};

// The output row encodes both the token and its reduction position. Production
// assigns consecutive top-k output rows per token, ordered by expert index.
struct RoutedRow {
    const char *expert;
    int token, output;
    float weight;
};

extern "C" int ds_recipe_group(void **, size_t *, const Job *, int, const RoutedRow *, int, const float *, float *);
