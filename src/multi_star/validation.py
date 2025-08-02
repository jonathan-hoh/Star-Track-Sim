def validate_triangle_matches(star_matches, ground_truth):
    """
    Validates the results of a 3-star triangle match.
    """
    if not star_matches:
        return {"status": "failed", "reason": "No matches found"}

    # For a 3-star test, we expect exactly one match
    if len(star_matches) != 1:
        return {"status": "failed", "reason": f"Expected 1 match, found {len(star_matches)}"}

    match = star_matches[0]
    expected_catalog = ground_truth['catalog']

    # Check if the matched catalog star is one of the expected stars
    if match.catalog_idx not in expected_catalog.index:
        return {"status": "failed", "reason": "Matched to an unexpected catalog star"}

    return {"status": "passed", "confidence": match.confidence}


def validate_pyramid_consistency(star_matches, ground_truth):
    """
    Validates the results of a 4-star pyramid match.
    """
    if len(star_matches) < 2:
        return {"status": "failed", "reason": f"Insufficient matches for pyramid validation, found {len(star_matches)}"}

    # All matched stars should belong to the same pyramid
    # In our simple case, they should all be from the same small catalog
    expected_catalog_indices = set(ground_truth['catalog'].index)
    matched_catalog_indices = {match.catalog_idx for match in star_matches}

    if not matched_catalog_indices.issubset(expected_catalog_indices):
        return {"status": "failed", "reason": "Matched stars from outside the expected pyramid"}

    # Further checks could be added here, e.g., on the consistency of the implied attitude
    return {"status": "passed", "matches_found": len(star_matches)}