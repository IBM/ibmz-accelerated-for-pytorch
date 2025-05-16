<!-- markdownlint-disable  MD033 -->

# Using the IBM Z Accelerated for PyTorch Container Image

# Table of contents

- [Overview](#overview)
- [Downloading the IBM Z Accelerated for PyTorch Container Image](#container)
- [Container Image Contents](#contents)
- [PyTorch Usage](#pytorch)
- [A Look into the Acceleration](#acceleration)
- [Security and Deployment Guidelines](#security-and-deployment-guidelines)
- [Execution on the Integrated Accelerator for AI and on CPU](#execution-paths)
- [Model Validation](#model-validation)
- [Using the Code Samples](#code-samples)
- [Frequently Asked Questions](#faq)
- [Technical Support](#contact)
- [Versioning Policy and Release Cadence](#versioning)
- [Licenses](#licenses)

# Overview <a id="overview"></a>

[PyTorch](https://pytorch.org/) is an open source machine learning
framework. It has a comprehensive set of tools that enable model development,
training, and inference. It also features a rich, robust ecosystem.

On IBM® z16™ and later (running Linux on IBM Z or IBM® z/OS® Container
Extensions (IBM zCX)), PyTorch will leverage new inference acceleration
capabilities that target the IBM Integrated Accelerator for AI through the
[IBM z Deep Neural Network](https://github.com/IBM/zDNN) (zDNN) library. The IBM
zDNN library contains a set of primitives that support Deep Neural Networks.
These primitives transparently target the IBM Integrated Accelerator for AI on
IBM z16 and later. No changes to the original model are needed to take advantage
of the new inference acceleration capabilities.

_Note. When using IBM Z Accelerated for PyTorch on either an IBM z15® or an
IBM z14®, PyTorch will transparently target the CPU with no changes to the
model._

# Downloading the IBM Z Accelerated for PyTorch Container Image <a id="container"></a>

Downloading the IBM Z Accelerated for PyTorch container image requires
credentials for the IBM Z and LinuxONE Container Registry,
[icr.io](https://icr.io).

Documentation on obtaining credentials to `icr.io` is located
[here](https://ibm.github.io/ibm-z-oss-hub/main/main.html).

---

Once credentials to `icr.io` are obtained and have been used to login to the
registry, you may pull (download) the IBM Z Accelerated for PyTorch container
image with the following code block:

```bash
# Replace X.X.X with the desired version to pull
docker pull icr.io/ibmz/ibmz-accelerated-for-pytorch:X.X.X
```

In the `docker pull` command illustrated above, the version specified above is
`X.X.X`. This is based on the version available in the
[IBM Z and LinuxONE Container Registry](https://ibm.github.io/ibm-z-oss-hub/containers/ibmz-accelerated-for-pytorch.html).
Release notes about a particular version can be found in this GitHub Repository
under releases
[here](https://github.com/IBM/ibmz-accelerated-for-pytorch/releases).

---

To remove the IBM Z Accelerated for PyTorch container image, please follow
the commands in the code block:

```bash
# Find the Image ID from the image listing
docker images

# Remove the image
docker rmi <IMAGE ID>
```

---

\*_Note. This documentation will refer to image/containerization commands in
terms of Docker. If you are utilizing Podman, please replace `docker` with
`podman` when using our example code snippets._

# Container Image Contents <a id="contents"></a>

To view a brief overview of the operating system version, software versions and
content installed in the container, as well as any release notes for each
released container image version, please visit the `releases` section of this
GitHub Repository, or you can click
[here](https://github.com/IBM/ibmz-accelerated-for-pytorch/releases).

# PyTorch Usage <a id="pytorch"></a>

For documentation on how to train and run inferences on models with PyTorch
please visit the official
[Open Source PyTorch documentation](https://pytorch.org/docs/stable/index.html).

For brief examples on how to run inferences on models with PyTorch please visit
our [samples section](#code-samples).

# A Look into the Acceleration <a id="acceleration"></a>

The acceleration is enabled through a Custom Device that get registered within
PyTorch.

- The registered Device will check PyTorch Ops for valid input(s) and/or
  output(s), targeting the accelerator where possible.
  - Only Ops with valid input(s) and/or output(s) will target the accelerator.
  - Some Ops with valid input(s) and/or output(s) may still not target the
    accelerator if their overhead is likely to outweigh any cost savings.

## Tensor

Ops will receive input(s) and/or output(s) in the form of `Tensor` objects.

- PyTorch's internal `Tensor` objects manage the shape, data type, and a
  pointer to the data buffer
- More info can be found [here](https://pytorch.org/docs/stable/tensors.html)

## Inference Mode Requirement

Model inference calls are currently only fully supported when used within an
[inference_mode](https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)
context, which has no interactions with autograd (e.g., model training).

Inference calls made outside a `inference_mode` context may not fully leverage
the accelerator.

## CPU Fallback

During runtime, input(s) and/or output(s) are checked to ensure they are the
correct shape and data type.

- If all shapes and data type are valid, the accelerator is used.
- If any shape or data type is invalid, the default CPU logic is used.

## NNPA Instruction Set Requirement

Before the Custom Device is registered, a call to `zdnn_is_nnpa_installed` is
made to ensure the NNPA instruction set for the accelerator is installed.

- If this call returns false, the Custom Device is not registered and runtime
  should proceed the same way PyTorch would without the acceleration
  benefits.

## Environment Variables for Logging <a id="env-variables"></a>

Certain environment variables can be set before execution to enable/disable
features or logs.

- `ZDNN_ENABLE_PRECHECK`: true

  - If set to true, zDNN will print logging information before running any
    computational operation.
  - Example: `export ZDNN_ENABLE_PRECHECK=true`.
    - Enable zDNN logging.

- `TORCH_NNPA_DEBUG`: 0 | 1

  - If set to 1, torch-nnpa will print logging information before running any
    operation.
  - Example: `export TORCH_NNPA_DEBUG=1`.
    - Enable torch-nnpa logging.

# Security and Deployment Guidelines <a id="security-and-deployment-guidelines"></a>

- For security and deployment best practices, please visit the common AI Toolkit
  documentation found
  [here](https://github.com/IBM/ai-toolkit-for-z-and-linuxone).

# Execution on the Integrated Accelerator for AI and on CPU <a id="execution-paths"></a>

## Execution Paths

IBM Z Accelerated for PyTorch follows IBM's train anywhere and deploy on IBM Z
strategy.

By default, when using the IBM Z Accelerated for PyTorch on an IBM z16 and later
system, PyTorch core will target the Integrated Accelerator for AI for a number
of compute-intensive operations during inferencing with no changes to the model.

It is common practice to write PyTorch code in a device-agnostic way,
and then switch between CPU and NNPA depending on what hardware is available.
Typically, to do this you might have used if-statements and ``nnpa()`` calls
to do this:

```python
import torch
import torch_nnpa

USE_NNPA = True

mod = torch.nn.Linear(20, 30)

# Sends `mod` from its device to NNPA using `nnpa()` call.
if USE_NNPA:
    mod.nnpa()

# Creates `inp` on NNPA by passing 'nnpa' as the device string.
device = 'nnpa' if USE_NNPA else 'cpu'
inp = torch.randn(128, 20, device=device)

mod.eval()

with torch.inference_mode():
    out = mod(inp)

# Copies `out` from its device to CPU.
# `cpu_out` will be on CPU.
# `out` will keep its original device.
cpu_out = out.to('cpu')

print(out.device)
print(cpu_out.device)

###################################################################
# PyTorch now also has a context manager which can take care of the
# device transfer automatically. Here is an example:

with torch.device('nnpa'):
    mod = torch.nn.Linear(20, 30)
    print(mod.weight.device)
    print(mod(torch.randn(128, 20)).device)
```

When using IBM Z Accelerated for PyTorch on either an IBM z15 or an IBM z14,
PyTorch will transparently target the CPU with no changes to the model.

## Training Workloads

Training workloads are currently not supported and may result in runtime errors.

It is recommended to perform training on the CPU. The saved model can then be
used for inference with IBM Z Accelerated for PyTorch.

## Scaled Dot Product Attention (SDPA) Backends

[Scaled Dot Product Attention](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
is in beta support for Pytorch and has multiple different backends.

We currently only support the
[SDPBackend.MATH](https://pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend)
backend, which can be used within a
[sdpa_kernel](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html#torch.nn.attention.sdpa_kernel)
context:

```python
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

...

with sdpa_kernel(SDPBackend.MATH):
    output = scaled_dot_product_attention(query, key, value)
```

# Model Validation <a id="model-validation"></a>

Various models that were trained on x86 or IBM Z have been validated to target
the IBM Integrated Accelerator for AI for a number of compute-intensive
operations during inferencing.

- [Image Classification](https://pytorch.org/vision/0.8/models.html#classification)
- [Semantic Segmentation](https://pytorch.org/vision/0.8/models.html#semantic-segmentation)
- [Object Detection](https://pytorch.org/vision/0.8/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)

_Note. Models that were trained outside of the PyTorch ecosystem may throw
endianness issues._

# Using the Code Samples <a id="code-samples"></a>

Documentation for our code samples can be found [here](samples).

# Frequently Asked Questions <a id="faq"></a>

## Q: Where can I get the IBM Z Accelerated for PyTorch container image?

Please visit this link
[here](https://ibm.github.io/ibm-z-oss-hub/containers/ibmz-accelerated-for-pytorch.html).
Or read the section titled
[Downloading the IBM Z Accelerated for PyTorch container image](#container).

## Q: Why are there multiple PyTorch container images in the IBM Z and LinuxONE Container Registry? <!-- markdownlint-disable-line MD013 -->

You may have seen multiple PyTorch container images in IBM Z and LinuxONE
Container Registry, namely
[ibmz/pytorch](https://ibm.github.io/ibm-z-oss-hub/containers/pytorch.html)
and
[ibmz/ibmz-accelerated-for-pytorch](https://ibm.github.io/ibm-z-oss-hub/containers/ibmz-accelerated-for-pytorch.html).

The `ibmz/pytorch` container image does not have support for the IBM
Integrated Accelerator for AI. The `ibmz/pytorch` container image only
transparently targets the CPU. It does not have any optimizations referenced in
this document.

The `ibmz/ibmz-accelerated-for-pytorch` container image includes support for
PyTorch core Graph Execution to transparently target the IBM Integrated
Accelerator for AI. The `ibmz/ibmz-accelerated-for-pytorch` container image
also still allows it's users to transparently target the CPU. This container
image contains the optimizations referenced in this document.

## Q: Where can I run the IBM Z Accelerated for PyTorch container image?

You may run the IBM Z Accelerated for PyTorch container image on IBM Linux on
Z or IBM® z/OS® Container Extensions (IBM zCX).

_Note. The IBM Z Accelerated for PyTorch container image will transparently
target the IBM Integrated Accelerator for AI on IBM z16 and later. However, if
using the IBM Z Accelerated for PyTorch container image on either an IBM z15
or an IBM z14, PyTorch will transparently target the CPU with no changes to
the model._

## Q: Can I install a newer or older version of PyTorch in the container?

No. Installing newer or older version of PyTorch than what is configured in
the container will not target the IBM Integrated Accelerator for AI.
Additionally, installing a newer or older version of PyTorch, or modifying
the existing PyTorch that is installed in the container image may have
unintended, unsupported, consequences. This is not advised.

# Technical Support <a id="contact"></a>

Information regarding technical support can be found
[here](https://github.com/IBM/ai-toolkit-for-z-and-linuxone).

# Versioning Policy and Release Cadence <a id="versioning"></a>

IBM Z Accelerated for PyTorch will follow the
[semantic versioning guidelines](https://semver.org/) with a few deviations.
Overall, IBM Z Accelerated for PyTorch follows a continuous release model
with a cadence of 1-2 minor releases per year. In general, bug fixes will be
applied to the next minor release and not back ported to prior major or minor
releases. Major version changes are not frequent and may include features
supporting new zSystems hardware as well as major feature changes in PyTorch
that are not likely backward compatible. Please refer to
[PyTorch guidelines](https://pytorch.org/get-started/previous-versions/) for
backwards compatibility across different versions of PyTorch.

## IBM Z Accelerated for PyTorch Versions

Each release version of IBM Z Accelerated for PyTorch has the form
MAJOR.MINOR.PATCH. For example, IBM Z Accelerated for PyTorch version 1.2.3
has MAJOR version 1, MINOR version 2, and PATCH version 3. Changes to each
number have the following meaning:

### MAJOR / VERSION

All releases with the same major version number will have API compatibility.
Major version numbers will remain stable. For instance, 1.X.Y may last 1 year or
more. It will potentially have backwards incompatible changes. Code and data
that worked with a previous major release will not necessarily work with the new
release.

### MINOR / FEATURE

Minor releases will typically contain new backward compatible features,
improvements, and bug fixes.

### PATCH / MAINTENANCE

Maintenance releases will occur more frequently and depend on specific patches
introduced (e.g. bug fixes) and their urgency. In general, these releases are
designed to patch bugs.

## Release cadence

Feature releases for IBM Z Accelerated for PyTorch occur about every 6 months
in general. Hence, IBM Z Accelerated for PyTorch 1.3.0 would generally be
released about 6 months after 1.2.0. Maintenance releases happen as needed in
between feature releases. Major releases do not happen according to a fixed
schedule.

# Licenses <a id="licenses"></a>

The International License Agreement for Non-Warranted Programs (ILAN) agreement
can be found
[here](https://www.ibm.com/support/customer/csol/terms/?id=L-NEYY-NZUN6H&lc=en).

The registered trademark Linux® is used pursuant to a sublicense from the Linux
Foundation, the exclusive licensee of Linus Torvalds, owner of the mark on a
worldwide basis.

PyTorch, the PyTorch logo and any related marks are trademarks of Facebook,
Inc.

Docker and the Docker logo are trademarks or registered trademarks of Docker,
Inc. in the United States and/or other countries. Docker, Inc. and other parties
may also have trademark rights in other terms used herein.

IBM, the IBM logo, and ibm.com, IBM z16, IBM z15, IBM z14 are trademarks or
registered trademarks of International Business Machines Corp., registered in
many jurisdictions worldwide. Other product and service names might be
trademarks of IBM or other companies. The current list of IBM trademarks can be
found [here](https://www.ibm.com/legal/copyright-trademark).
