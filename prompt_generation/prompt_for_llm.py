prompt = '''You are given two inputs:
      1. "domain = {domain}"
      2. "classname = {classname}"

      Instructions:
      1. First, interpret the domain to better understand the context or category of the classname. For example:
        - If domain is "animals", then classname is a type of animal.
        - If domain is "vehicles", then classname is a type of vehicle.
        - etc.
        
      2. Determine if the classname, within the given domain, is an animate being (something that moves on its own) or an inanimate object.
        - Animate examples: animals, humans, robots that move
        - Inanimate examples: furniture, plants (if considered stationary), tools, etc.

      3. Based on the domain and classname, produce four lists in **valid JSON** format (no extra text or explanation, just valid JSON):
        - "action_or_pose": 3 to 5 possible actions or poses
          • If it's animate, list dynamic actions or poses (e.g., "running", "jumping").
          • If it's inanimate, list states or positions (e.g., "standing upright", "stacked", "lying flat").
        - "invariant_features": 3 to 5 short, **unique** fine-grained features that belong to this classname and **MUST** not be found in other classes
          • Keep each feature concise (e.g., "striped fur", "spiral shell", "red tail light").
        - "cooccurrence": 3 to 5 objects or entities that commonly appear with or near this classname, within the given domain context.
        - "background": 3 to 5 background settings typically associated with this classname.

      4. Output must be JSON. 
        - The output keys must be exactly: "action_or_pose", "invariant_features", "cooccurrence", "background".
        - Each value must be an array of strings.
        - No extra keys or explanation should appear beyond those four arrays.

      5. Provide the final result in this exact JSON structure:

      {{
        "action_or_pose": ["...","...","..."],
        "invariant_features": ["...","...","..."],
        "cooccurrence": ["...","...","..."],
        "background": ["...","...","..."]
      }}'''