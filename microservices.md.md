Building microservices in Python involves breaking a large, monolithic application into smaller, loosely coupled services. Each service handles a **single business capability**, runs in its own process, and communicates with other services using lightweight protocols (like HTTP REST or message queues).

To design robust microservices, we rely on core patterns: API Gateways, Service-to-Service communication, and Event-Driven architecture.

---

### The Core Architecture Pattern

Before looking at the code, it's important to understand how these services interact. In a production environment, clients don't talk to individual microservices directly; instead, they go through an API Gateway, and the microservices talk to each other or a shared message broker behind the scenes.

---

### 📦 Sample Project: E-Commerce Microservices

Let's build a mini e-commerce system split into two independent microservices:

1. **Product Service (FastAPI):** Manages items and inventory.
2. **Order Service (FastAPI + `httpx`):** Handles customer purchases and checks with the Product Service to verify inventory before confirming an order.

---

### Service 1: The Product Service

This service runs independently, managing its own data store (simulated here in memory).

Create a file named `product_service.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Product Service")

# Simulated Database
PRODUCTS = {
    101: {"name": "Wireless Mouse", "price": 29.99, "stock": 5},
    102: {"name": "Mechanical Keyboard", "price": 89.99, "stock": 2},
}

class Product(BaseModel):
    name: str
    price: float
    stock: int

@app.get("/products/{product_id}")
def get_product(product_id: int):
    if product_id not in PRODUCTS:
        raise HTTPException(status_code=404, detail="Product not found")
    return PRODUCTS[product_id]

@app.post("/products/{product_id}/deduct")
def deduct_stock(product_id: int, quantity: int = 1):
    if product_id not in PRODUCTS:
        raise HTTPException(status_code=404, detail="Product not found")
    if PRODUCTS[product_id]["stock"] < quantity:
        raise HTTPException(status_code=400, detail="Insufficient stock")
    
    PRODUCTS[product_id]["stock"] -= quantity
    return {"message": "Stock updated", "remaining_stock": PRODUCTS[product_id]["stock"]}

if __name__ == "__main__":
    import uvicorn
    # Run this service on port 8001
    uvicorn.run(app, host="127.0.0.1", port=8001)

```

---

### Service 2: The Order Service

This service orchestrates purchases. It must perform synchronous **Service-to-Service communication** by calling the Product Service via an HTTP request before finalizing an order.

Create a file named `order_service.py`:

```python
from fastapi import FastAPI, HTTPException
import httpx

app = FastAPI(title="Order Service")

# URL of our independent Product Service
PRODUCT_SERVICE_URL = "http://127.0.0.1:8001"

@app.post("/orders/")
async def create_order(product_id: int, quantity: int):
    async with httpx.AsyncClient() as client:
        # 1. Ask Product Service if the item exists and check details
        product_resp = await client.get(f"{PRODUCT_SERVICE_URL}/products/{product_id}")
        
        if product_resp.status_code == 404:
            raise HTTPException(status_code=404, detail="Cannot place order. Product does not exist.")
        
        product_data = product_resp.json()
        
        # 2. Check stock locally based on service response
        if product_data["stock"] < quantity:
            raise HTTPException(status_code=400, detail="Cannot place order. Out of stock.")
            
        # 3. Request Product Service to deduct the stock
        deduct_resp = await client.post(
            f"{PRODUCT_SERVICE_URL}/products/{product_id}/deduct?quantity={quantity}"
        )
        
        if deduct_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to reserve inventory.")

        # 4. Process order success logic
        total_price = product_data["price"] * quantity
        return {
            "status": "Order Placed Successfully",
            "product_name": product_data["name"],
            "total_cost": round(total_price, 2)
        }

if __name__ == "__main__":
    import uvicorn
    # Run this service on port 8002
    uvicorn.run(app, host="127.0.0.1", port=8002)

```

---

### How to Run and Test This Project

To experience how these services function independently, follow these steps:

1. **Install dependencies:**
```bash
pip install fastapi uvicorn httpx

```


2. **Start Product Service:** Open a terminal and run `python product_service.py`. It starts listening on port `8001`.
3. **Start Order Service:** Open a second terminal window and run `python order_service.py`. It starts listening on port `8002`.
4. **Test an Order Execution:** Open your browser or use a terminal curl command to try placing an order via port `8002`:
```bash
curl -X POST "http://127.0.0.1:8002/orders/?product_id=101&quantity=2"

```


**The result:** The Order Service will ping the Product Service, verify the item info, confirm stock, deduct it on port `8001`, and return a success message back to you!

---

### Production-Level Python Tools for Microservices

While the example above uses direct HTTP calls, real-world Python microservices handle scale using a specialized ecosystem:

* **Asynchronous Communication (Message Brokers):** If the Product Service goes down in our example, orders fail. In production, tools like **RabbitMQ** or **Apache Kafka** (using Python libraries like `aiormq` or `confluent-kafka`) allow services to exchange events asynchronously. The Order service emits an `OrderPlaced` event, and the inventory service consumes it when ready.
* **Containerization (Docker):** Every microservice is packaged into its own Docker container with its own dependencies, ensuring it runs exactly the same way in local environments as it does in production clouds.
* **Service Mesh & Orchestration:** **Kubernetes** is typically used to manage, scale, and route traffic to hundreds of Python app containers seamlessly.

## SECURITY

text
Securing microservices is uniquely challenging because you are moving from a monolithic setup (where everything happens securely in memory on a single machine) to a distributed network where services constantly whisper to each other over open wires.

To build bulletproof security, architects divide traffic into two categories: **North-South traffic** (clients connecting to your system) and **East-West traffic** (microservices talking to one another).

---

### 1. Perimeter Security (North-South Traffic)

You never want to expose your individual microservices directly to the public internet. Instead, all external requests enter through a secure boundary.

* **The API Gateway:** This acts as the single point of entry (the "bouncer" of your architecture). It handles SSL termination, rate limiting, and initial authentication.
* **OAuth2 and OpenID Connect (OIDC):** When a user logs in, an Identity Provider (like Auth0, Keycloak, or AWS Cognito) issues a **JWT (JSON Web Token)**.
* **Token Verification:** The client sends this JWT with every request. The API Gateway verifies the token's signature. If it's valid, the gateway passes the request inward.

---

### 2. Service-to-Service Security (East-West Traffic)

Once a request is inside your network, you cannot assume the internal network is safe. This is where the **Zero Trust Architecture** principle comes in: *Never trust, always verify.*

#### A. Mutual TLS (mTLS) for Network-Level Security

In standard HTTPS, the client verifies the identity of the server. In **mTLS**, *both* parties verify each other using cryptographic certificates.

* When Service A calls Service B, they exchange TLS certificates.
* This ensures that communication is completely **encrypted** (no one can sniff the data) and **authenticated** (Service B knows the caller is genuinely Service A, not an attacker masquerading on the network).

#### B. JWT Propagation for Application-Level Security

While mTLS proves *which machine* is talking, it doesn't prove *which user* authorized the action. For that, services pass user context along the chain.

* The API Gateway forwards the user's JWT down to Service A.
* If Service A needs to call Service B, it forwards (propagates) that same JWT.
* Service B can then check: *"Is this user actually allowed to delete this record?"*

---

### 3. The Modern Approach: Using a Service Mesh

Managing mTLS certificates, rotations, and security logic inside your actual Python or Go code is a nightmare for developers. To solve this, production environments use a **Service Mesh** (like Istio, Linkerd, or Consul).

A Service Mesh uses a **Sidecar Pattern**. Every microservice gets a tiny, dedicated network proxy running right next to it.

* **How it works:** Your Python code thinks it is making a plain, unencrypted HTTP request to a local port. The sidecar proxy intercepts that traffic, automatically upgrades it to an encrypted **mTLS** connection, handles the handshake with the receiving service's sidecar, and passes it through safely.
* This strips all security logic out of your application code and offloads it entirely to the infrastructure layer.

---

### 4. Authorization & Network Policies

Knowing *who* someone is (Authentication) isn't the same as knowing what they are *allowed to do* (Authorization).

* **Role-Based Access Control (RBAC):** You can define fine-grained policies. For example: "The *Billing Service* is allowed to read from the *Order Service*, but the *Front-End Web Service* is strictly blocked from talking to the *Billing Service* directly."
* **Kubernetes Network Policies:** At the firewall/network layer, you can configure your cluster so that containers housing non-essential services physically cannot route TCP packets to your highly sensitive database or payment containers.

## APIGEE
When you implement microservice security using **Google Cloud Apigee**, you are leveraging an enterprise-grade API Management platform to handle the **Perimeter Security (North-South)** layer and seamlessly bridge it with your **Internal Security (East-West)** layer.

Apigee sits in front of your microservices, acting as an intelligent, secure reverse proxy. Here is exactly how security is built around microservices using Apigee policies and architecture.

---

### 1. Perimeter Security: Edge Protection

When an external client requests data, Apigee is the first wall they hit. It handles authentication before traffic ever reaches your cloud network or Kubernetes cluster.

* **OAuth2 / OIDC Policies:** Apigee has built-in `OAuthV2` policies. It can act as the OAuth server itself (generating, validating, and refreshing access tokens) or interface with an external identity provider (like Okta or Ping Identity) to validate incoming JSON Web Tokens (JWTs).
* **Threat Mitigation:** Apigee enforces security at the boundary using out-of-the-box policies:
* **Spike Arrest & Rate Limiting:** Prevents Denial of Service (DoS) attacks from overwhelming your internal microservices.
* **JSON/XML Threat Protection:** Inspects incoming payloads to block SQL injection, oversized arrays, or malicious deep-nesting structures.



---

### 2. Token Exchange & Context Propagation

A major security problem in microservices is **Token Bloat**. External clients pass an opaque OAuth token, but your internal Python or Go microservices need to know the specific user ID, roles, and permissions.

Apigee solves this using **Token Translation**:

```text
[External Client] ──( Opaque Access Token )──> [ Apigee Edge ]
                                                       │
                                            (Verify & Transform)
                                                       ▼
[Internal Microservice] <──( Secure Signed JWT )───────┘

```

1. The client sends a generic, opaque OAuth token to Apigee.
2. Apigee's `VerifyAccessToken` policy checks it.
3. If valid, Apigee uses a `GenerateJWT` policy to mint a short-lived, cryptographically signed **Internal JWT**. This internal token contains the user's cleared identity and claims.
4. Apigee injects this new JWT into the HTTP header and forwards it to your internal microservice network.

---

### 3. Securing the South-to-East Bridge (Apigee to Microservices)

Once Apigee validates a request, it must safely hand it off to your backend microservices. Apigee secures this connection using two primary methods:

#### A. Northbound/Southbound mTLS

Apigee uses **Keystores and Truststores** to manage TLS certificates.

* To secure the backend handoff, you configure a **TargetEndpoint** in Apigee with `SSLInfo` enabled.
* Apigee presents its client certificate to your microservice gateway (like an Ingress Controller), and your gateway presents its server certificate to Apigee. This enforces **Mutual TLS (mTLS)** at the entry point of your internal cluster.

#### B. Apigee Adapter for Service Mesh (Istio)

If your microservices are running inside Kubernetes using a service mesh like Istio, Google provides the **Apigee Adapter for Istio**.

Instead of routing all internal service-to-service traffic back up to the main Apigee cloud, the Apigee adapter runs locally as a plug-in inside your mesh's control plane.

* When **Service A** tries to talk to **Service B** internally, the local Envoy sidecar proxy intercepts the call and asks the local Apigee adapter if the operation is secure.
* The adapter applies Apigee API keys, quota checks, and RBAC rules instantly at microsecond speeds, giving you Apigee-level security *inside* your internal microservice mesh without added network latency.
  
