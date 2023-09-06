# Object Relation for 3D Object Detection

<img src="./seen_in_context.jpeg" width="100%">

3D object detectors usually predict objects independently from each other. Given the limited receptive field for each object, they are missing the required context. Especially when it comes to heavily occoluded or sparse objects, context is necessary to correctly predict 3D bounding boxes. Simply increasing the receptive field would not give enough context while still being computational feasible. Therefore, this project, comparable to previous work, proposes to model context efficiently with object relation.

<figure>
  <img src="./object_relation.png" width="100%">
  <figcaption>Object Relation as an efficient Way to model Context</figcaption>
</figure>


