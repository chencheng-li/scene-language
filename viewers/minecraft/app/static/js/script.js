var deepslateResources;
const { mat4, vec3 } = glMatrix;

function upperPowerOfTwo(x) {
  x -= 1;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 18;
  x |= x >> 32;
  return x + 1;
}

document.addEventListener("DOMContentLoaded", function (event) {
  const image = document.getElementById("atlas");
  if (image.complete) {
    loadResources(image);
  } else {
    image.addEventListener("load", () => loadResources(image));
  }
});

// Taken from Deepslate examples
function loadResources(textureImage) {
  const blockDefinitions = {};
  Object.keys(assets.blockstates).forEach((id) => {
    blockDefinitions["minecraft:" + id] = deepslate.BlockDefinition.fromJson(
      id,
      assets.blockstates[id]
    );
  });

  const blockModels = {};
  Object.keys(assets.models).forEach((id) => {
    blockModels["minecraft:" + id] = deepslate.BlockModel.fromJson(
      id,
      assets.models[id]
    );
  });
  Object.values(blockModels).forEach((m) =>
    m.flatten({ getBlockModel: (id) => blockModels[id] })
  );

  const atlasCanvas = document.createElement("canvas");
  const atlasSize = upperPowerOfTwo(
    textureImage.width >= textureImage.height
      ? textureImage.width
      : textureImage.height
  );
  atlasCanvas.width = textureImage.width;
  atlasCanvas.height = textureImage.height;
  const atlasCtx = atlasCanvas.getContext("2d");
  atlasCtx.drawImage(textureImage, 0, 0);
  const atlasData = atlasCtx.getImageData(0, 0, atlasSize, atlasSize);
  const idMap = {};
  Object.keys(assets.textures).forEach((id) => {
    const [u, v, du, dv] = assets.textures[id];
    const dv2 = du !== dv && id.startsWith("block/") ? du : dv;
    idMap["minecraft:" + id] = [
      u / atlasSize,
      v / atlasSize,
      (u + du) / atlasSize,
      (v + dv2) / atlasSize,
    ];
  });

  const textureAtlas = new deepslate.TextureAtlas(atlasData, idMap);

  deepslateResources = {
    getBlockDefinition(id) {
      return blockDefinitions[id];
    },
    getBlockModel(id) {
      return blockModels[id];
    },
    getTextureUV(id) {
      return textureAtlas.getTextureUV(id);
    },
    getTextureAtlas() {
      return textureAtlas.getTextureAtlas();
    },
    getBlockFlags(id) {
      return { opaque: false };
    },
    getBlockProperties(id) {
      return null;
    },
    getDefaultBlockProperties(id) {
      return null;
    },
  };
}

function createRenderer(structure) {
  // Create canvas and size it appropriately
  // TODO: Make size change on window resize
  const viewer = document.getElementById("viewer");
  const canvas = document.createElement("canvas");
  viewer.appendChild(canvas);

  canvas.width =
    window.innerWidth ||
    document.documentElement.clientWidth ||
    document.body.clientWidth;
  canvas.height =
    window.innerHeight ||
    document.documentElement.clientHeight ||
    document.body.clientHeight;

  // Remove old content
  const oldContent = document.getElementById("main-content");
  oldContent.style.display = "none";

  // Create Deepslate Renderer
  // Need chunksize 8 as seems to be a max number of faces per chunk that will render
  const gl = canvas.getContext("webgl");
  const renderer = new deepslate.StructureRenderer(
    gl,
    structure,
    deepslateResources,
    (options = { chunkSize: 8 })
  );

  // Crappy controls
  let viewDist = 4;
  let xRotation = 0.8;
  let yRotation = 0.5;
  let xOffset = 0;
  let yOffset = 0;
  const size = structure.getSize();
  let cameraPos = vec3.create();
  vec3.set(cameraPos, -size[0] / 2, -size[1] / 2, -size[2] / 2);

  // refactor this code to use separate functions for each type of control
  function render() {
    yRotation = yRotation % (Math.PI * 2);
    xRotation = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, xRotation));
    viewDist = Math.max(1, Math.min(20, viewDist));

    const view = mat4.create();
    mat4.rotateX(view, view, xRotation);
    mat4.rotateY(view, view, yRotation);
    mat4.translate(view, view, cameraPos); //[xOffset, yOffset, -viewDist]);
    //mat4.translate(view, view, );

    renderer.drawStructure(view);
    renderer.drawGrid(view);
  }

  requestAnimationFrame(render);

  function move3d(direction, relativeVertical = true, sensitivity = 1) {
    let offset = vec3.create();
    vec3.set(
      offset,
      direction[0] * sensitivity,
      direction[1] * sensitivity,
      direction[2] * sensitivity
    );
    if (relativeVertical) {
      vec3.rotateX(offset, offset, [0, 0, 0], -xRotation * sensitivity);
    }
    vec3.rotateY(offset, offset, [0, 0, 0], -yRotation * sensitivity);
    vec3.add(cameraPos, cameraPos, offset);
  }

  function pan(direction, sensitivity = 1) {
    // seems backwards but is correct
    yRotation += (direction[0] / 200) * sensitivity;
    xRotation += (direction[1] / 200) * sensitivity;
  }

  function runMovementFunction(evt, sensitivitySetting = null) {
    let sensitivity = 1;
    if (sensitivitySetting) {
      sensitivity *= parseFloat(localStorage.getItem(sensitivitySetting) ?? 1);
    }
    pan(evt, sensitivity);
    // controls[value](evt);
  }

  let clickPos = null;
  canvas.addEventListener("mousedown", (evt) => {
    evt.preventDefault();
    clickPos = [evt.clientX, evt.clientY];
  });
  canvas.addEventListener("mousemove", (evt) => {
    if (clickPos) {
      const args = [evt.clientX - clickPos[0], evt.clientY - clickPos[1]];
      runMovementFunction(args, "click-drag-sensitivity");
      clickPos = [evt.clientX, evt.clientY];
      requestAnimationFrame(render);
    }
  });

  canvas.addEventListener("mouseup", (evt) => {
    clickPos = null;
  });
  canvas.addEventListener("mouseout", (evt) => {
    clickPos = null;
  });
  canvas.addEventListener("wheel", (evt) => {
    evt.preventDefault();
    move3d([0, 0, -evt.deltaY / 200]);
    requestAnimationFrame(render);
  });

  const moveDist = 0.2;
  const keyMoves = {
    w: [0, 0, moveDist],
    s: [0, 0, -moveDist],
    a: [moveDist, 0, 0],
    d: [-moveDist, 0, 0],
    ArrowUp: [0, 0, moveDist],
    ArrowDown: [0, 0, -moveDist],
    ArrowLeft: [moveDist, 0, 0],
    ArrowRight: [-moveDist, 0, 0],
    Shift: [0, moveDist, 0],
    " ": [0, -moveDist, 0],
  };
  let pressedKeys = new Set();
  let animationFrameId;

  document.addEventListener("keydown", (evt) => {
    const key = evt.key.toLowerCase(); // Convert to lowercase
    if (keyMoves[key] || evt.key === "Shift") {
      evt.preventDefault();
      pressedKeys.add(key);
      startAnimation();
    }
  });

  document.addEventListener("keyup", (evt) => {
    const key = evt.key.toLowerCase(); // Convert to lowercase
    if (keyMoves[key] || evt.key === "Shift") {
      evt.preventDefault();
      pressedKeys.delete(key);
      stopAnimationIfNeeded();
    }
  });

  // Reset key states when window loses focus to prevent stuck keys
  window.addEventListener("blur", () => {
    pressedKeys.clear();
    stopAnimationIfNeeded();
  });

  function startAnimation() {
    if (!animationFrameId) {
      animationFrameId = requestAnimationFrame(updatePosition);
    }
  }

  function stopAnimationIfNeeded() {
    if (pressedKeys.size === 0 && animationFrameId) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }
  }

  function updatePosition() {
    if (pressedKeys.size === 0) {
      stopAnimationIfNeeded();
      return;
    }

    let direction = vec3.create();
    for (const key of pressedKeys) {
      if (keyMoves[key]) {
        vec3.add(direction, direction, keyMoves[key]);
      }
    }

    // Apply vertical movement only if Shift is pressed
    if (pressedKeys.has("shift")) {
      vec3.add(direction, direction, keyMoves["Shift"]);
    }

    move3d(direction, false);
    render();

    // Continue the animation loop
    animationFrameId = requestAnimationFrame(updatePosition);
  }
}

function structureFromJsonData(data) {
  const blocks = data.blocks;
  const structure = new deepslate.Structure([
    data.width,
    data.height,
    data.depth,
  ]);

  // Map to hold 3D coordinates as keys and block type and properties as values
  const blockMap = new Map();

  // Function to generate a unique key for each 3D coordinate
  const generateKey = (x, y, z) => `${x},${y},${z}`;

  // Iterate over blocks and populate the blockMap
  blocks.forEach((block) => {
    const { start, end, type, properties, fill } = block;
    const [startX, startY, startZ] = start;
    const [endX, endY, endZ] = end;

    for (let x = startX; x < endX; x++) {
      for (let y = startY; y < endY; y++) {
        for (let z = startZ; z < endZ; z++) {
          if (
            fill ||
            x === startX ||
            x === endX - 1 ||
            y === startY ||
            y === endY - 1 ||
            z === startZ ||
            z === endZ - 1
          ) {
            const key = generateKey(x, y, z);
            blockMap.set(key, { type, properties });
          }
        }
      }
    }
  });

  // Counter for the number of blocks added
  let blockCount = 0;

  // Iterate over the blockMap and add blocks to the structure
  blockMap.forEach((value, key) => {
    const [x, y, z] = key.split(",").map(Number);
    try {
      if (value.properties) {
        structure.addBlock([x, y, z], value.type, value.properties);
      } else {
        structure.addBlock([x, y, z], value.type);
      }
      blockCount++;
    } catch (err) {
      console.warn(
        `Was unable to add block of type ${value.type} at position ${x}, ${y}, ${z}.`
      );
    }
  });

  console.log("Done!", blockCount, "blocks created");

  return structure;
}

/* Set the width of the side navigation to 250px */
function openSettings() {
  document.getElementById("settings-panel").style.width = "800px";
}

/* Set the width of the side navigation to 0 */
function closeSettings() {
  document.getElementById("settings-panel").style.width = "0";
}
