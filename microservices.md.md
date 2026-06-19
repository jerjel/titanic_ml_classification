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