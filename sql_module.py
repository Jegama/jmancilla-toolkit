from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, PickleType, create_engine
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Formatter(Base):
    __tablename__ = 'formatters'

    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True, nullable=False)
    history = Column(PickleType, nullable=False)

# Create an engine that stores data in the local directory's
# sqlalchemy_example.db file.
engine = create_engine('sqlite:///database.db', connect_args={'check_same_thread': False})

# Create all tables in the engine. This is equivalent to "Create Table"
# statements in raw SQL.
Base.metadata.create_all(engine)

# Create a new session
Session = sessionmaker(bind=engine)